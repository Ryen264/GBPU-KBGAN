import os
import logging
import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
from typing import Generator, Tuple

from datasets import batch_by_num, BernCorrupterMulti, BernCorrupter
from models import TransE, TransD, DistMult, ComplEx
from metrics import classification_metrics
import config

class Component():
    def __init__(self, role: str, model_type: str, n_entity: int, n_relation: int):
        """
        role = ["discriminator", "generator"]
        model_type = ["TransE", "TransD", "DistMult", "ComplEx"]
        """
        if role in ["discriminator", "generator"]:
            print(f"Initialized a new component with role {role}.")
            self.role = role
        else:
            raise ValueError(f"Input role should be in list [\"discriminator\", \"generator\"]!")
        
        if model_type in ["TransE", "TransD", "DistMult", "ComplEx"]:
            print(f'Initialized component: {model_type} model.')
            self.model_type = model_type
        else:
            raise ValueError(f"Input model type should be in list [\"TransE\", \"TransD\", \"DistMult\", \"ComplEx\"]!")

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.model_type = model_type
        self.model_path = None

        self.model_config = config._config[self.model_type]

        use_gpu = (config.device.type == 'cuda')
        if self.model_type == 'TransE':
            self.model = TransE(self.n_entity, self.n_relation, use_gpu=use_gpu)
        elif self.model_type == 'TransD':
            self.model = TransD(self.n_entity, self.n_relation, use_gpu=use_gpu)
        elif self.model_type == 'DistMult':
            self.model = DistMult(self.n_entity, self.n_relation, use_gpu=use_gpu)
        elif self.model_type == 'ComplEx':
            self.model = ComplEx(self.n_entity, self.n_relation, use_gpu=use_gpu)    

    def load(self, model_path: str) -> None:
        print(f"Loading component: {self.model_type} model.")
        self.model.load_model(model_path)
        self.model_path = model_path
        print(f"Loaded component successfully by: {self.model_path}")

    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
              use_early_stopping: bool=False, patience: int=10, optimizer_name: str='Adam',
              is_save_model: bool=True) -> Tuple[float, str | None]:    
        config.overwrite_config_with_args(["--log.prefix=" + self.model_type + '_'])
        config.logger_init()

        if self.model_type in ['TransE', 'TransD']:
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
        elif self.model_type in ['DistMult', 'ComplEx']:
            self.n_sample = self.model_config.n_sample
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if config._config.task == 'triple-classification' or config._config.task == 'all':
            tester = lambda: classification_metrics(*self.model.evaluate_triple_classification(valid_data, self.n_entity, heads, tails))
        elif config._config.task == 'link-prediction' or config._config.task == 'all':
            tester = lambda: self.model.evaluate(valid_data, self.n_entity, heads, tails)
        else:
            raise ValueError(f"Unsupported task: {config._config.task}")
        use_gpu = (config.device.type == 'cuda')

        print(f'Training component: {self.model_type} model.')
        best_perf, model_path = self.model.train(train_data, corrupter, tester,
                                                use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
                                                use_gpu=use_gpu, is_save_model=is_save_model)
        print(f'Trained component successfully: {self.model_type} model.')
        if is_save_model:
            self.model_path = model_path
            return best_perf, model_path
        return best_perf, None
    
    def save(self) -> str:
        if self.model_path is None:
            raise ValueError("Component must be fitted before being saved!")
        
        print(f"Saving component: {self.model_type} model.")
        self.model_path = self.model.save()
        print(f"Saved component successfully by: {self.model_path}")

    def get_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.get_score(head, relation, tail)

    def generator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                        n_sample: int=1, temperature: float=1.0, train: bool=True) -> Generator[torch.Tensor, torch.Tensor, None]:
        """
        Generator step: sample fake triples and update with REINFORCE
        """
        if (self.role != "generator"):
            raise ValueError("This component is not a generator!")
        if not self.model.is_trained_or_loaded():
            raise ValueError("Generator must be pretrained or loaded before generator step!")

        # Forward pass: generate samples
        n, m = tail.size()
        relation_var = Variable(relation.to(config.device))
        head_var = Variable(head.to(config.device))
        tail_var = Variable(tail.to(config.device))

        logits = self.model.get_prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits, dim=-1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]
        
        # Yield samples to get rewards from discriminator
        rewards = yield sample_heads, sample_tails
        
        # Backward pass: update generator with REINFORCE
        if train:            
            self.model._ensure_optimizer()
            self.model.model.zero_grad()

            log_probs = nnf.log_softmax(logits, dim=-1)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.to(config.device), sample_idx.data])
            reinforce_loss.backward()

            self.model.opt.step()
            self.model.model.constraint()
        yield None

    def discriminator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                            head_fake: torch.Tensor, tail_fake: torch.Tensor, train: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminator step: distinguish real from fake triples
        """
        if (self.role != "discriminator"):
            raise ValueError("This component is not a discriminator!")
        if not self.model.is_trained_or_loaded():
            raise ValueError("Discriminator must be pretrained or loaded before discriminator step!")
        
        # Forward pass: compute losses and scores
        head_var = Variable(head.to(config.device))
        relation_var = Variable(relation.to(config.device))
        tail_var = Variable(tail.to(config.device))
        
        head_fake_var = Variable(head_fake.to(config.device))
        tail_fake_var = Variable(tail_fake.to(config.device))
        
        losses = self.model.model.pair_loss(head_var, relation_var, tail_var, head_fake_var, tail_fake_var)
        fake_scores = self.model.model.score(head_fake_var, relation_var, tail_fake_var)
                
        # Backward pass: update discriminator
        if train:
            self.model._ensure_optimizer()
            self.model.model.zero_grad()

            torch.sum(losses).backward()

            self.model.opt.step()
            self.model.model.constraint()
        return losses.data, -fake_scores.data

    def evaluate_on_ranking(self, test_data: tuple, heads: torch.Tensor, tails: torch.Tensor,
                            filt=True, k_list=[1, 3, 10]) -> dict:
        if not self.model.is_trained_or_loaded():
            raise ValueError("Component must be trained before being tested!")
        
        print(f"Testing component on task ranking: {self.model_type} model.")
        metrics = self.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails,
                                                 filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_classification(self, test_data: tuple, optimizing_metric='accuracy') -> dict:
        if not self.model.is_trained_or_loaded():
            raise ValueError("Component must be trained before being tested!")
        
        print(f"Testing component on task classification: {self.model_type} model.")
        metrics = self.model.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)
        return metrics

class KBGAN():
    def __init__(self, discriminator_type: str, generator_type: str,
                 n_entity: int, n_relation: int):
        """
        discriminator_type = ["TransE", "TransD"]
        generator_type = ["DistMult", "ComplEx"]
        """
        self.discriminator_type = discriminator_type
        self.discriminator = Component(role="discriminator", model_type=discriminator_type,
                                       n_entity=n_entity, n_relation=n_relation)

        self.generator_type = generator_type
        self.generator = Component(role="generator", model_type=generator_type,
                                   n_entity=n_entity, n_relation=n_relation)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.discriminator_path = None
        self.generator_path = None

        task_dir = '.\\models\\' + config._config.dataset + '\\' + config._config.task
        os.makedirs(task_dir, exist_ok=True)
        model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        self.kbgan_path = os.path.join(task_dir, model_name)

        self.n_sample = config._config.KBGAN.n_sample
        self.temperature = config._config.KBGAN.temperature
        self.n_epoch = config._config.KBGAN.n_epoch
        self.n_batch = config._config.KBGAN.n_batch

    def load_discriminator(self, discriminator_path: str=None) -> None:
        print(f"Loading discriminator: {self.discriminator_type} model.")
        self.discriminator.load(discriminator_path)
        self.discriminator_path = discriminator_path
        print(f"Loaded discriminator successfully by path: {discriminator_path}")

    def load_generator(self, generator_path: str=None) -> None:
        print(f"Loading generator: {self.generator_type} model.")
        self.generator.load(generator_path)
        self.generator_path = generator_path
        print(f"Loaded generator successfully by path: {generator_path}")

    def load_kbgan(self, kbgan_path: str) -> None:      
        print(f"Loading KBGAN (discriminator)...")
        self.discriminator.load(kbgan_path)
        self.kbgan_path = kbgan_path
        print(f"Loaded KBGAN (discriminator) successfully by: {self.kbgan_path}")

    def save_kbgan(self) -> str: 
        if self.kbgan_path is None:
            raise ValueError("KBGAN path is not set. Cannot save model.")
              
        print(f"Saving KBGAN (discriminator)...")
        self.kbgan_path = self.discriminator.save()
        print(f"Saved KBGAN (discriminator) successfully to: {self.kbgan_path}")

    def train_components(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                use_early_stopping: bool=False, patience: int=10, optimizer_name: str='Adam',
                is_save_components: bool=True) -> Tuple[float, str | None, float, str | None]:        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        
        config.overwrite_config_with_args(["--log.prefix=" + self.discriminator_type + '-' + self.generator_type + "_"])
        config.logger_init()

        print(f"Training KBGAN's components: {self.generator_type} generator, {self.discriminator_type} discriminator.")
        print(f"Training discriminator...")
        best_perf_d, path_d = self.discriminator.train(heads, tails, train_data, valid_data,
                                                    use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
                                                    is_save_model=is_save_components)
        if is_save_components and path_d is not None:
            self.discriminator_path = path_d
            print(f"Trained discriminator is saved to: {path_d}")

        print(f"Training generator...")
        best_perf_g, path_g = self.generator.train(heads, tails, train_data, valid_data,
                                                    use_early_stopping=use_early_stopping, patience=patience, optimizer_name=optimizer_name,
                                                    is_save_model=is_save_components)
        if is_save_components and path_g is not None:
            self.generator_path = path_g
            print(f"Trained generator is saved to: {path_g}.")
        return best_perf_d, path_d, best_perf_g, path_g
           
    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
                use_early_stopping: bool=False, patience: int=10, optimizer_name: str = 'Adam',
                is_save_kbgan: bool=True) -> Tuple[float, str]:
        if (not self.generator.model.is_trained_or_loaded()) or (not self.discriminator.model.is_trained_or_loaded()):
            raise ValueError("Both generator and discriminator must be pretrained or loaded before being trained!")
        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data[0], torch.Tensor):
            valid_data = [torch.LongTensor(vec) for vec in valid_data]
        
        config.overwrite_config_with_args(["--log.prefix=" + self.discriminator_type + '-' + self.generator_type + "_"])
        config.logger_init()

        # Initialize optimizers according to optimizer_name for both models
        opt_map = {
            'Adam': Adam,
            'SGD': SGD,
            'AdamW': AdamW,
            'RMSprop': RMSprop,
            'Adagrad': Adagrad,
        }
        opt_cls = opt_map.get(optimizer_name, Adam)
        try:
            self.generator.model.opt = opt_cls(self.generator.model.parameters())
            self.discriminator.model.opt = opt_cls(self.discriminator.model.parameters())
        except Exception:
            pass

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
                gen_step = self.generator.generator_step(hs, rs, ts, temperature=self.temperature)
                head_smpl, tail_smpl = next(gen_step)
                
                losses, rewards = self.discriminator.discriminator_step(h, r, t, head_fake=head_smpl.squeeze(), tail_fake=tail_smpl.squeeze())
                epoch_reward += torch.sum(rewards)

                rewards = rewards - avg_reward

                # Update generator with rewards
                try:
                    gen_step.send(rewards.unsqueeze(1))
                except StopIteration:
                    pass
                epoch_d_loss += torch.sum(losses)
                
            avg_loss = epoch_d_loss / n_train
            avg_reward = epoch_reward / n_train

            logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, self.n_epoch, avg_loss, avg_reward)

            if (epoch + 1) % config().KBGAN.epoch_per_test == 0:
                metrics = self.discriminator.model.evaluate(valid_data, self.n_entity, heads, tails, filt=True)
                perf = metrics['MRR']
                if perf > best_perf:
                    if is_save_kbgan:
                        print(f"Saving KBGAN at epoch {epoch + 1} with MRR {best_perf}.")
                        self.kbgan_path = self.save_kbgan()
                        print(f"Saved KBGAN successfully to: {self. kbgan_path}")
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
                                        filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if (not self.discriminator.model.is_trained_or_loaded()):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics

    def evaluate_kbgan_on_triple_classification(self, test_data_with_labels: tuple, optimizing_metric='accuracy') -> Tuple[dict, float, list, list]:
        if (not self.discriminator.model.is_trained_or_loaded()):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        
        print("Evaluating KBGAN discriminator on Triple Classification...")
        metrics = self.discriminator.evaluate_on_classification(test_data_with_labels, optimizing_metric=optimizing_metric)
        return metrics

    def evaluate_discriminator_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if (not self.discriminator.model.is_trained_or_loaded()):
            raise ValueError("Discriminator must be trained before being tested!")
    
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating discriminator on Link Prediction...")
        metrics = self.discriminator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics
    
    def evaluate_generator_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if (not self.generator.model.is_trained_or_loaded()):
            raise ValueError("Generator must be trained before being tested!")
        
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating generator on Link Prediction...")
        metrics = self.generator.model.evaluate_on_ranking(test_data, self.n_entity, heads, tails, filt=filt, k_list=k_list)
        return metrics
    
    def evaluate_discriminator_on_triple_classification(self, test_data: tuple, optimizing_metric='accuracy') -> dict:
        if (not self.discriminator.model.is_trained_or_loaded()):
            raise ValueError("Discriminator must be trained before being tested!")
        
        print("Evaluating discriminator on Triple Classification...")
        metrics = self.discriminator.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)
        return metrics
    
    def evaluate_generator_on_triple_classification(self, test_data: tuple, optimizing_metric='accuracy') -> dict:
        if (not self.generator.model.is_trained_or_loaded()):
            raise ValueError("Generator must be trained before being tested!")
        
        print("Evaluating generator on Triple Classification...")
        metrics = self.generator.evaluate_on_classification(test_data, optimizing_metric=optimizing_metric)
        return metrics