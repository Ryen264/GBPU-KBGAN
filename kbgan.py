import os
import logging
from typing import Generator, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad

from datasets import batch_by_num, BernCorrupterMulti, BernCorrupter
from models import TransE, TransD, DistMult, ComplEx
import config

class Component:
    def __init__(self, role: str, model_type: str, n_entity: int, n_relation: int):
        valid_roles = {"discriminator", "generator"}
        valid_models = {"TransE", "TransD", "DistMult", "ComplEx"}
        if role not in valid_roles:
            raise ValueError(f"Input role should be in {valid_roles}!")
        if model_type not in valid_models:
            raise ValueError(f"Input model type should be in {valid_models}!")
        print(f"Initialized a new component with role {role} and model {model_type}.")
        self.role = role
        self.model_type = model_type
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.model_config = config._config[model_type]
        use_gpu = (config.device.type == 'cuda')
        model_cls = {"TransE": TransE, "TransD": TransD, "DistMult": DistMult, "ComplEx": ComplEx}[model_type]
        self.model = model_cls(n_entity, n_relation, use_gpu=use_gpu)
        self.model_path: Optional[str] = None
        self.n_sample = getattr(self.model_config, 'n_sample', None)

    def load(self, model_path: str) -> None:
        if self.n_entity is None or self.n_relation is None:
            raise ValueError("Component must be fitted before being loaded!")
        print(f"Loading component by path: {model_path}")
        model_cls = {"TransE": TransE, "TransD": TransD, "DistMult": DistMult, "ComplEx": ComplEx}[self.model_type]
        self.model = model_cls(self.n_entity, self.n_relation, use_gpu=(config.device.type == 'cuda'))
        self.model.load(model_path)
        self.model_path = model_path
        print(f"Loaded component successfully by: {self.model_path}")

    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
              use_early_stopping: bool = False, patience: int = 10, optimizer_name: str = 'Adam',
              is_save_model: bool = True) -> Tuple[float, Optional[str]]:
        config.overwrite_config_with_args([f"--log.prefix={self.model_type}_"])
        config.logger_init()
        if self.model_type in {"TransE", "TransD"}:
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
        elif self.model_type in {"DistMult", "ComplEx"}:
            self.n_sample = self.model_config.n_sample
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        if config._config.task in {"triple-classification", "all"}:
            tester = lambda: self.model.evaluate_on_classification(valid_data, optimizing_metric='accuracy')
        elif config._config.task in {"link-prediction", "all"}:
            tester = lambda: self.model.evaluate_on_ranking(valid_data, self.n_entity, heads, tails, filt=True, k_list=[1, 3, 10])
        else:
            raise ValueError(f"Unsupported task: {config._config.task}")
        use_gpu = (config.device.type == 'cuda')
        print(f'Training component: {self.model_type} model.')
        best_perf, model_path = self.model.train(
            train_data, corrupter, tester,
            use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
            use_gpu=use_gpu, is_save_model=is_save_model)
        print(f'Trained component successfully: {self.model_type} model.')
        if is_save_model:
            self.model_path = model_path
            return best_perf, model_path
        return best_perf, None
    
    def save(self, model_path: Optional[str] = None) -> str:
        if model_path is not None:
            self.model_path = model_path
        if self.model_path is None:
            raise ValueError("Component must be fitted before being saved!")
        print(f"Saving component: {self.model_type} model.")
        self.model_path = self.model.save(self.model_path)
        print(f"Saved component successfully by: {self.model_path}")
        return self.model_path

    def get_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.get_score(head, relation, tail)

    def generator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                      n_sample: int = 1, temperature: float = 1.0, train: bool = True) -> Generator[torch.Tensor, torch.Tensor, None]:
        """Generator step: sample fake triples and update with REINFORCE."""
        if self.role != "generator":
            raise ValueError("This component is not a generator!")
        if not self.model:
            raise ValueError("Generator must be pretrained or loaded before generator step!")
        n, m = tail.size()
        device = self.model.device
        head_var = Variable(head.to(device))
        relation_var = Variable(relation.to(device))
        tail_var = Variable(tail.to(device))
        logits = self.model.model.prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits, dim=-1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]
        rewards = yield sample_heads, sample_tails
        if train:
            self.model.model.zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.to(device), sample_idx.data])
            reinforce_loss.backward()
            self.model.opt.step()
            self.model.model.constraint()
        yield None

    def discriminator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                          head_fake: torch.Tensor, tail_fake: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminator step: distinguish real from fake triples."""
        if self.role != "discriminator":
            raise ValueError("This component is not a discriminator!")
        if not self.model:
            raise ValueError("Discriminator must be pretrained or loaded before discriminator step!")
        device = self.model.device
        head_var = Variable(head.to(device))
        relation_var = Variable(relation.to(device))
        tail_var = Variable(tail.to(device))
        head_fake_var = Variable(head_fake.to(device))
        tail_fake_var = Variable(tail_fake.to(device))
        losses = self.model.model.pair_loss(head_var, relation_var, tail_var, head_fake_var, tail_fake_var)
        fake_scores = self.model.model.score(head_fake_var, relation_var, tail_fake_var)
        if train:
            self.model.model.zero_grad()
            torch.sum(losses).backward()
            self.model.opt.step()
            self.model.model.constraint()
        return losses.data, -fake_scores.data

    def evaluate_on_ranking(self, test_data: tuple, heads: torch.Tensor, tails: torch.Tensor,
                           filt: bool = True, k_list: Optional[list] = None) -> dict:
        if k_list is None:
            k_list = [1, 3, 10]
        if not self.model:
            raise ValueError("Component must be trained before being tested!")
        print(f"Testing component on task ranking: {self.model_type} model.")
        return self.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)

    def evaluate_on_classification(self, test_data: tuple, optimizing_metric: str = 'accuracy') -> dict:
        if not self.model:
            raise ValueError("Component must be trained before being tested!")
        print(f"Testing component on task classification: {self.model_type} model.")
        return self.model.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)


class KBGAN:
    def __init__(self, discriminator_type: str, generator_type: str,
                 n_entity: int, n_relation: int):
        """
        discriminator_type = ["TransE", "TransD"]
        generator_type = ["DistMult", "ComplEx"]
        """
        self.discriminator_type = discriminator_type
        self.generator_type = generator_type
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.discriminator = Component("discriminator", discriminator_type, n_entity, n_relation)
        self.generator = Component("generator", generator_type, n_entity, n_relation)
        self.discriminator_path: Optional[str] = None
        self.generator_path: Optional[str] = None
        task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
        os.makedirs(task_dir, exist_ok=True)
        model_name = f'kbgan_dis-{self.discriminator_type}_gen-{self.generator_type}.mdl'
        self.kbgan_path = os.path.join(task_dir, model_name)
        self.n_sample = config._config.KBGAN.n_sample
        self.temperature = config._config.KBGAN.temperature
        self.n_epoch = config._config.KBGAN.n_epoch
        self.n_batch = config._config.KBGAN.n_batch

    def load_discriminator(self, discriminator_path: Optional[str] = None) -> None:
        print(f"Loading discriminator: {self.discriminator_type} model.")
        self.discriminator.load(discriminator_path)
        self.discriminator_path = discriminator_path
        print(f"Loaded discriminator successfully by path: {discriminator_path}")

    def load_generator(self, generator_path: Optional[str] = None) -> None:
        print(f"Loading generator: {self.generator_type} model.")
        self.generator.load(generator_path)
        self.generator_path = generator_path
        print(f"Loaded generator successfully by path: {generator_path}")

    def load_kbgan(self, kbgan_path: str) -> None:
        print(f"Loading KBGAN (discriminator)...")
        self.discriminator.load(kbgan_path)
        self.kbgan_path = kbgan_path
        print(f"Loaded KBGAN (discriminator) successfully by: {self.kbgan_path}")

    def save_kbgan(self, save_path: Optional[str] = None) -> str:
        if save_path is not None:
            self.kbgan_path = save_path
        if self.kbgan_path is None:
            task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
            os.makedirs(task_dir, exist_ok=True)
            model_name = f'kbgan_dis-{self.discriminator_type}_gen-{self.generator_type}.mdl'
            self.kbgan_path = os.path.join(task_dir, model_name)
        print(f"Saving KBGAN (discriminator)...")
        self.kbgan_path = self.discriminator.save(self.kbgan_path)
        print(f"Saved KBGAN (discriminator) successfully to: {self.kbgan_path}")
        return self.kbgan_path

    def train_components(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                        use_early_stopping: bool = False, patience: int = 10, optimizer_name: str = 'Adam',
                        is_save_components: bool = True) -> Tuple[float, Optional[str], float, Optional[str]]:
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        print(f"Training KBGAN's components: {self.generator_type} generator, {self.discriminator_type} discriminator.")
        print(f"Training discriminator...")
        best_perf_d, path_d = self.discriminator.train(
            heads, tails, train_data, valid_data,
            use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
            is_save_model=is_save_components)
        if is_save_components and path_d is not None:
            self.discriminator_path = path_d
            print(f"Trained discriminator is saved to: {path_d}")
        print(f"Training generator...")
        best_perf_g, path_g = self.generator.train(
            heads, tails, train_data, valid_data,
            use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
            is_save_model=is_save_components)
        if is_save_components and path_g is not None:
            self.generator_path = path_g
            print(f"Trained generator is saved to: {path_g}.")
        return best_perf_d, path_d, best_perf_g, path_g
           
    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                    use_early_stopping: bool = False, patience: int = 10, optimizer_name: str = 'Adam',
                    is_save_kbgan: bool = True) -> Tuple[float, Optional[str]]:
        if (not self.generator.model) or (not self.discriminator.model):
            raise ValueError("Both generator and discriminator must be pretrained or loaded before being trained!")
        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]

        # log_vars[0] for Ranking, log_vars[1] for Classification
        self.log_vars = torch.nn.Parameter(torch.zeros(2, requires_grad=True, device=self.discriminator.model.device))
        # Initialize optimizers
        opt_map = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}
        opt_cls = opt_map.get(optimizer_name, Adam)
        try:
            self.generator.model.opt = opt_cls(self.generator.model.model.parameters())
            self.discriminator.model.opt = opt_cls(list(self.discriminator.model.model.parameters()) + [self.log_vars])
        except (AttributeError, TypeError):
            pass
        bce_criterion = torch.nn.BCELoss()

        corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.n_sample)
        head, relation, tail = train_data
        n_train = len(head)

        best_perf = 0.0
        avg_reward = 0.0
        patience_counter = 0

        print(f'Training KBGAN: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        for epoch in range(self.n_epoch):
            epoch_d_loss = 0
            epoch_reward = 0
            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(self.n_batch, head, relation, tail, head_cand, relation_cand, tail_cand, n_sample=n_train):
                h = h.to(self.discriminator.model.device)
                r = r.to(self.discriminator.model.device)
                t = t.to(self.discriminator.model.device)
                gen_step = self.generator.generator_step(hs, rs, ts, temperature=self.temperature)
                head_smpl, tail_smpl = next(gen_step)
                loss_rank, rewards = self.discriminator.discriminator_step(h, r, t, head_fake=head_smpl.squeeze(), tail_fake=tail_smpl.squeeze(), train=True)
                pos_scores = self.discriminator.model.model(h, r, t)
                neg_scores = self.discriminator.model.model(head_smpl.squeeze().to(self.discriminator.model.device), r, tail_smpl.squeeze().to(self.discriminator.model.device))
                pos_scores_norm = torch.clamp(torch.sigmoid(pos_scores), min=1e-6, max=1.0-1e-6).float()
                neg_scores_norm = torch.clamp(torch.sigmoid(neg_scores), min=1e-6, max=1.0-1e-6).float()
                target_pos = torch.ones_like(pos_scores_norm, dtype=torch.float32)
                target_neg = torch.zeros_like(neg_scores_norm, dtype=torch.float32)
                loss_class = bce_criterion(pos_scores_norm, target_pos) + bce_criterion(neg_scores_norm, target_neg)
                loss_rank_scalar = torch.mean(loss_rank)
                precision_rank = torch.exp(-self.log_vars[0])
                loss_rank_weighted = precision_rank * loss_rank_scalar + self.log_vars[0]
                precision_class = torch.exp(-self.log_vars[1])
                loss_class_weighted = precision_class * loss_class + self.log_vars[1]
                total_loss = 5.0 * loss_rank_weighted + loss_class_weighted
                self.discriminator.model.opt.zero_grad()
                total_loss.backward()
                self.discriminator.model.opt.step()
                self.discriminator.model.model.constraint()
                epoch_reward += torch.sum(rewards)
                epoch_d_loss += total_loss.item()
                rewards = rewards - avg_reward
                try:
                    gen_step.send(rewards.unsqueeze(1))
                except StopIteration:
                    pass
            avg_loss = epoch_d_loss / n_train
            avg_reward = epoch_reward / n_train
            logging.info('Epoch %d/%d, Joint_Loss=%f, Rank_W=%f, Class_W=%f',
                         epoch + 1, self.n_epoch, avg_loss,
                         self.log_vars[0].item(), self.log_vars[1].item())
            if (epoch + 1) % config._config.KBGAN.epoch_per_test == 0:
                metrics = self.discriminator.model.evaluate_on_ranking(valid_data, self.n_entity, heads, tails, filt=True)
                perf = metrics['MRR']
                if perf > best_perf:
                    if is_save_kbgan:
                        print(f"Saving KBGAN at epoch {epoch + 1} with MRR {best_perf}.")
                        self.kbgan_path = self.save_kbgan()
                        print(f"Saved KBGAN successfully to: {self.kbgan_path}")
                    best_perf = perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                if use_early_stopping and patience_counter >= patience:
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, patience)
                    break
        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        if is_save_kbgan:
            print(f"Saving trained KBGAN (discriminator) with best MRR {best_perf}.")
            self.kbgan_path = self.save_kbgan()
            print(f"Saved trained KBGAN (discriminator) successfully to: {self.kbgan_path}")
            return best_perf, self.kbgan_path
        return best_perf, None

    def evaluate_kbgan_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list=None) -> dict:
        if k_list is None:
            k_list = [1, 3, 10]
        if (not self.discriminator.model):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics

    def evaluate_kbgan_on_triple_classification(self, test_data_with_labels: tuple, optimizing_metric='accuracy') -> Tuple[dict, float, list, list]:
        if (not self.discriminator.model):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        
        print("Evaluating KBGAN discriminator on Triple Classification...")
        metrics = self.discriminator.evaluate_on_classification(test_data_with_labels, optimizing_metric=optimizing_metric)
        return metrics

    def evaluate_discriminator_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list=None) -> dict:
        if k_list is None:
            k_list = [1, 3, 10]
        if (not self.discriminator.model):
            raise ValueError("Discriminator must be trained before being tested!")
    
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating discriminator on Link Prediction...")
        metrics = self.discriminator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics
    
    def evaluate_generator_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list=None) -> dict:
        if k_list is None:
            k_list = [1, 3, 10]
        if (not self.generator.model):
            raise ValueError("Generator must be trained before being tested!")
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating generator on Link Prediction...")
        metrics = self.generator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics
    
    def evaluate_discriminator_on_triple_classification(self, test_data: tuple, optimizing_metric='accuracy') -> dict:
        if (not self.discriminator.model):
            raise ValueError("Discriminator must be trained before being tested!")
        
        print("Evaluating discriminator on Triple Classification...")
        metrics = self.discriminator.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)
        return metrics
    
    def evaluate_generator_on_triple_classification(self, test_data: tuple, optimizing_metric='accuracy') -> dict:
        if (not self.generator.model):
            raise ValueError("Generator must be trained before being tested!")
        
        print("Evaluating generator on Triple Classification...")
        metrics = self.generator.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)
        return metrics
