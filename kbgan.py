import os
import logging
import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
from typing import Generator, Tuple, Optional
import numpy as np

from datasets import batch_by_num, batch_by_size, BernCorrupterMulti, BernCorrupter
from models import TransE, TransD, DistMult, ComplEx
import config
import metrics

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
        if (self.n_entity is None or self.n_relation is None):
            raise ValueError("Component must be fitted before being loaded!")

        print(f"Loading component by path: {model_path}")
        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation, self.model_config)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation, self.model_config)
        self.model.load(model_path)
        self.model_path = model_path
        print(f"Loaded component successfully by: {self.model_path}")

    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data: tuple,
              rank_class_balance: float = 5.0, use_early_stopping: bool=False, patience: int=10,
              optimizer_name: str='Adam', is_save_model: bool=True) -> Tuple[float, Optional[str]]:    
        config.overwrite_config_with_args(["--log.prefix=" + self.model_type + '_'])
        config.logger_init()

        if self.model_type in ['TransE', 'TransD']:
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
        elif self.model_type in ['DistMult', 'ComplEx']:
            self.n_sample = self.model_config.n_sample
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        class_metrics = lambda: self.evaluate_on_classification(valid_data, optimizing_metric='accuracy', threshold=None)
        rank_metrics = lambda: self.evaluate_on_ranking(valid_data, heads, tails, filt=True, k_list=None)
        tester = lambda: (rank_class_balance * rank_metrics()['mrr'] + class_metrics()['accuracy']) / (rank_class_balance + 1)

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
    
    def save(self, model_path: Optional[str] = None) -> str:
        """Persist underlying model; allow overriding the destination path."""
        if model_path is not None:
            self.model_path = model_path

        if self.model_path is None:
            raise ValueError("Component must be fitted before being saved!")

        print(f"Saving component: {self.model_type} model.")
        self.model_path = self.model.save(self.model_path)
        print(f"Saved component successfully by: {self.model_path}")
        return self.model_path

    def opt_zero_grad(self) -> None:
        self.model.ensure_optimizer()
        self.model.opt.zero_grad()

    def opt_step(self) -> None:
        self.model.opt.step()
        self.model.constraint()

    def set_optimizer(self, optimizer_name: str) -> None:
        opt_map = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}
        opt_cls = opt_map.get(optimizer_name, Adam)
        try:
            self.model.opt = opt_cls(self.model.parameters())
        except (AttributeError, TypeError):
            pass

    def get_score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.get_score(head, relation, tail)
    
    def get_device(self) -> torch.device:
        return self.model.device

    def is_trained_or_loaded(self) -> bool:
        return self.model.is_trained_or_loaded()

    def is_distance_based(self) -> bool:
        """Check if model is distance-based (lower score is better)."""
        return self.model_type in ['TransE', 'TransD']

    def generator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                        n_sample: int=1, temperature: float=1.0, train: bool=True) -> Generator[torch.Tensor, torch.Tensor, None]:
        """
        Generator step: sample fake triples and update with REINFORCE
        """
        if (self.role != "generator"):
            raise ValueError("This component is not a generator!")
        if not self.is_trained_or_loaded():
            raise ValueError("Generator must be pretrained or loaded before generator step!")

        # Forward pass: generate samples
        n, m = tail.size()
        model_device = self.get_device()

        relation_var, head_var, tail_var = Variable(relation.to(model_device)), Variable(head.to(model_device)), Variable(tail.to(model_device))

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
            self.opt_zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.to(model_device), sample_idx.data])
            reinforce_loss.backward()
            self.opt_step()
        yield None
       
    def discriminator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                            head_fake: torch.Tensor, tail_fake: torch.Tensor, train: bool=True) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Discriminator step: distinguish real from fake triples
        """
        if (self.role != "discriminator"):
            raise ValueError("This component is not a discriminator!")
        if not self.is_trained_or_loaded():
            raise ValueError("Discriminator must be pretrained or loaded before discriminator step!")

        def calculate_d(head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:            
            return self.model.model.dist(head, relation, tail)

        model_device = self.get_device()

        # Forward pass: compute losses and scores
        head_var, relation_var, tail_var = Variable(head.to(model_device)), Variable(relation.to(model_device)), Variable(tail.to(model_device))        
        head_fake_var, tail_fake_var = Variable(head_fake.to(model_device)), Variable(tail_fake.to(model_device))
        
        d_good = calculate_d(head_var, relation_var, tail_var)
        d_bad = calculate_d(head_fake_var, relation_var, tail_fake_var)
        pair_loss = nnf.relu(d_good - d_bad + self.model_config.margin)
        fake_scores = self.get_score(head_fake_var, relation_var, tail_fake_var)
                
        # Backward pass: update discriminator
        if train:
            self.opt_zero_grad()
            sum_loss = torch.sum(pair_loss)
            sum_loss.backward()
            self.opt_step()
        return pair_loss.data, -fake_scores.data, d_good.data.max().item(), d_bad.data.min().item()

    def evaluate_on_ranking(self, test_data: tuple, heads: torch.Tensor, tails: torch.Tensor,
                            filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if not self.is_trained_or_loaded():
            raise ValueError("Component must be trained before being tested!")

        model_device = self.get_device()

        print(f"Testing component on task ranking: {self.model_type} model.")
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        test_data_no_label = test_data[:3]
        count = 0
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(config._config.test_batch_size, *test_data_no_label):
                batch_size = batch_head.size(0)

                all_var = torch.arange(0, self.n_entity).unsqueeze(0).expand(batch_size, self.n_entity).long().to(model_device)
                head_var = batch_head.unsqueeze(1).expand(batch_size, self.n_entity).to(model_device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, self.n_entity).to(model_device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, self.n_entity).to(model_device)

                batch_head_scores = self.get_score(all_var, relation_var, tail_var)
                batch_tail_scores = self.get_score(head_var, relation_var, all_var)
            
                batch_head_scores = batch_head_scores.detach()
                batch_tail_scores = batch_tail_scores.detach()

                for head, relation, tail, head_scores, tail_scores in zip(batch_head, batch_relation, batch_tail, batch_head_scores, batch_tail_scores):
                    head_id, relation_id, tail_id = head.item(), relation.item(), tail.item()
                    if filt:
                        key_head = (tail_id, relation_id)
                        if key_head in heads and heads[key_head]._nnz() > 1:
                            tmp = head_scores[head_id].item()
                            head_scores += heads[key_head].to(model_device) * 1e30
                            head_scores[head_id] = tmp
                            
                        key_tail = (head_id, relation_id)
                        if key_tail in tails and tails[key_tail]._nnz() > 1:
                            tmp = tail_scores[tail_id].item()
                            tail_scores += tails[key_tail].to(model_device) * 1e30
                            tail_scores[tail_id] = tmp

                    head_metrics = metrics.ranking_metrics(scores=head_scores, target=head_id, k_list=k_list)
                    tail_metrics = metrics.ranking_metrics(scores=tail_scores, target=tail_id, k_list=k_list)

                    head_mr, head_mrr, head_hits = head_metrics['mr'], head_metrics['mrr'], head_metrics['hits']
                    tail_mr, tail_mrr, tail_hits = tail_metrics['mr'], tail_metrics['mrr'], tail_metrics['hits']

                    mr_total += (head_mr + tail_mr)
                    mrr_total += (head_mrr + tail_mrr)
                    hits_total = [(hits_total[i] + head_hits[i] + tail_hits[i]) for i in range(len(k_list))]
                    count += 2
                    
        mr_rate = mr_total / count
        mrr_rate = mrr_total / count
        hits_rate = [hit_total / count for hit_total in hits_total]
        
        ranking_metrics = {}
        ranking_metrics['mr'] = mr_rate
        ranking_metrics['mrr'] = mrr_rate
        for i in range(len(k_list)):
            ranking_metrics[f'hit@{k_list[i]}'] = hits_rate[i]

        ranking_metrics_str = f"Ranking metrics: {ranking_metrics}\n"
        logging.info(ranking_metrics_str)
        return ranking_metrics

    def evaluate_on_classification(self, test_data: tuple, optimizing_metric: str='accuracy', threshold: float=None) -> dict:
        if not self.is_trained_or_loaded():
            raise ValueError("Component must be trained before being tested!")
        
        model_device = self.get_device()
        print(f"Testing component on task classification: {self.model_type} model.")
        
        def find_optimal_threshold(valid_data: tuple, labels: list, n_thresholds: int=100) -> float:
            """
            Find the optimal threshold for triple classification using validation data.

            Args:
                valid_data: Tuple of (heads, relations, tails)
                labels: Ground truth labels for validation data
                n_thresholds: Number of threshold values to try

            Returns:
                Optimal threshold value that maximizes F1 score
            """
            heads, relations, tails = valid_data

            # Compute scores for all validation samples (batched for efficiency)
            scores_list = []
            with torch.no_grad():
                for batch_head, batch_relation, batch_tail in batch_by_size(config._config.test_batch_size,
                                                                           heads, relations, tails):
                    head_var = torch.LongTensor(batch_head).to(model_device)
                    relation_var = torch.LongTensor(batch_relation).to(model_device)
                    tail_var = torch.LongTensor(batch_tail).to(model_device)

                    batch_scores = self.get_score(head_var, relation_var, tail_var)
                    batch_scores = batch_scores.detach().cpu().numpy()
                    scores_list.extend(batch_scores.tolist())

            # Try different threshold values
            scores_array = np.array(scores_list)
            min_score = float(scores_array.min())
            max_score = float(scores_array.max())
            threshold_values = np.linspace(min_score, max_score, n_thresholds)

            best_val = 0.0
            best_threshold = 0.0

            # Determine if model is distance-based or similarity-based
            is_distance_based = self.is_distance_based()

            for threshold in threshold_values:
                if is_distance_based:
                    predictions = np.where(scores_array < threshold, 1, 0).tolist()
                else:
                    predictions = np.where(scores_array > threshold, 1, 0).tolist()
                
                classification_metrics = metrics.classification_metrics(predictions, labels, scores=scores_list)
                val_metric = classification_metrics.get(optimizing_metric, 0.0)
                
                if val_metric > best_val:
                    best_val = val_metric
                    best_threshold = threshold

            logging.info(f"Optimal threshold: {best_threshold:.4f} ({optimizing_metric}={best_val:.4f})")
            return best_threshold

        if len(test_data) < 4:
            raise ValueError("For classification metrics, test_data must include labels as the 4th element (heads, relations, tails, labels).")

        heads_list, relations_list, tails_list, labels = test_data
        scores_list = []
        true_labels = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(config._config.test_batch_size,
                                                                                     heads_list, relations_list, tails_list, labels):
                # ensure tensors on device
                head_var = torch.LongTensor(batch_head).to(model_device)
                relation_var = torch.LongTensor(batch_relation).to(model_device)
                tail_var = torch.LongTensor(batch_tail).to(model_device)

                batch_scores = self.get_score(head_var, relation_var, tail_var)
                batch_scores = batch_scores.detach().cpu().tolist()

                scores_list.extend([float(s) for s in batch_scores])
                true_labels.extend([int(x) for x in batch_label])

        if len(scores_list) == 0:
            raise ValueError("No samples found in test_data for classification evaluation.")

        if threshold is None:
            n_thresholds = 100
            threshold = find_optimal_threshold(valid_data=(heads_list, relations_list, tails_list), labels=true_labels, n_thresholds=n_thresholds)
            logging.info(f"Determined optimal threshold for classification: {threshold:.4f} using validation data with {n_thresholds} thresholds.")

        # determine whether smaller score means positive (distance-based models)
        is_distance_based = self.is_distance_based()

        # Vectorized prediction generation
        scores_array = np.array(scores_list)
        if is_distance_based:
            predictions = np.where(scores_array < threshold, 1, 0).tolist()
        else:
            predictions = np.where(scores_array > threshold, 1, 0).tolist()

        # For distance-based models lower scores mean better; invert for AUC so higher is better
        scores_for_auc = [-s for s in scores_list] if is_distance_based else scores_list

        classification_metrics = metrics.classification_metrics(predictions, true_labels, scores=scores_for_auc)
        # Format metrics for cleaner output
        classification_metrics_display = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in classification_metrics.items()}
        classification_metrics_str = f"Classification metrics: {classification_metrics_display}\n"
        logging.info(classification_metrics_str)
        return classification_metrics

class KBGAN():
    def __init__(self, discriminator_type: str, generator_type: str, n_entity: int, n_relation: int):
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

        task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
        os.makedirs(task_dir, exist_ok=True)
        model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        self.kbgan_path = os.path.join(task_dir, model_name)

        self.n_sample = config._config.KBGAN.n_sample
        self.temperature = config._config.KBGAN.temperature
        self.n_epoch = config._config.KBGAN.n_epoch
        self.n_batch = config._config.KBGAN.n_batch

        self.max_d_good = 0.0
        self.min_d_bad = float('inf')

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

    def save_kbgan(self, save_path: Optional[str] = None) -> str:
        """Save discriminator parameters to the KBGAN path."""
        if save_path is not None:
            self.kbgan_path = save_path
        if self.kbgan_path is None:
            # Construct path if not already set
            task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
            os.makedirs(task_dir, exist_ok=True)
            model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
            self.kbgan_path = os.path.join(task_dir, model_name)

        print(f"Saving KBGAN (discriminator)...")
        self.kbgan_path = self.discriminator.save(self.kbgan_path)
        print(f"Saved KBGAN (discriminator) successfully to: {self.kbgan_path}")
        return self.kbgan_path

    def train_components(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                rank_class_balance: float=5.0, use_early_stopping: bool=False, patience: int=10,
                optimizer_name: str='Adam', is_save_components: bool=True) -> Tuple[float, Optional[str], float, Optional[str]]:        
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        print(f"Training KBGAN's components: {self.generator_type} generator, {self.discriminator_type} discriminator.")
        print(f"Training discriminator...")
        best_perf_d, path_d = self.discriminator.train(heads, tails, train_data, valid_data_w_label,
                                                    rank_class_balance, use_early_stopping=use_early_stopping, patience=patience,
                                                    optimizer_name=optimizer_name, is_save_model=is_save_components)
        if is_save_components and path_d is not None:
            self.discriminator_path = path_d
            print(f"Trained discriminator is saved to: {path_d}")

        print(f"Training generator...")
        best_perf_g, path_g = self.generator.train(heads, tails, train_data, valid_data_w_label,
                                                    rank_class_balance, use_early_stopping=use_early_stopping, patience=patience,
                                                    optimizer_name=optimizer_name, is_save_model=is_save_components)
        if is_save_components and path_g is not None:
            self.generator_path = path_g
            print(f"Trained generator is saved to: {path_g}.")
        return best_perf_d, path_d, best_perf_g, path_g
           
    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                rank_class_balance: float=5.0, use_early_stopping: bool=False, patience: int=10,
                optimizer_name: str='Adam', is_save_kbgan: bool=True) -> Tuple[float, str]:
        if (not self.generator.is_trained_or_loaded()) or (not self.discriminator.is_trained_or_loaded()):
            raise ValueError("Both generator and discriminator must be pretrained or loaded before being trained!")
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        model_device = self.discriminator.get_device()

        # log_vars[0] for Ranking, log_vars[1] for Classification
        # Initializing at 0 means initial weight sigma=1
        self.log_vars = torch.nn.Parameter(torch.zeros(2, requires_grad=True, device=model_device)) # Ensure device matches model

        # Initialize optimizers according to optimizer_name for both models
        self.discriminator.set_optimizer(optimizer_name)
        self.generator.set_optimizer(optimizer_name)

        # Define Classification Loss function
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
                # Move tensors to device
                h = h.to(model_device)
                r = r.to(model_device)
                t = t.to(model_device)
                
                # --- Generator Step ---
                gen_step = self.generator.generator_step(hs, rs, ts, temperature=self.temperature)
                head_smpl, tail_smpl = next(gen_step)
                
                # --- Discriminator Step ---
                # 1. Get Ranking Loss (and rewards for Generator)
                loss_rank, rewards, max_d_good, min_d_bad = self.discriminator.discriminator_step(h, r, t, head_fake=head_smpl.squeeze(), tail_fake=tail_smpl.squeeze(), train=True)

                # Update max_d_good and min_d_bad for monitoring
                self.max_d_good = max(self.max_d_good, max_d_good)
                self.min_d_bad = min(self.min_d_bad, min_d_bad)

                # 2. Get Classification Loss (BCE)
                # We need raw scores from the discriminator to compute BCE.
                pos_scores = self.discriminator.model.model(h, r, t)
                neg_scores = self.discriminator.model.model(head_smpl.squeeze().to(model_device), r, tail_smpl.squeeze().to(model_device))
                
                # Normalize scores to [0, 1] using sigmoid and clamp to avoid numerical issues
                pos_scores_norm = torch.clamp(torch.sigmoid(pos_scores), min=1e-6, max=1.0-1e-6).float()
                neg_scores_norm = torch.clamp(torch.sigmoid(neg_scores), min=1e-6, max=1.0-1e-6).float()
                
                # Target: 1 for Real, 0 for Fake
                target_pos = torch.ones_like(pos_scores_norm, dtype=torch.float32)
                target_neg = torch.zeros_like(neg_scores_norm, dtype=torch.float32)

                # Formula: L_total = exp(-s1)*L1 + s1 + exp(-s2)*L2 + s2
                # L1 = Ranking Loss, L2 = Classification Loss, s1 and s2 are log_vars for dynamic weighting

                # Calculate classification loss
                loss_class = bce_criterion(pos_scores_norm, target_pos) + bce_criterion(neg_scores_norm, target_neg)

                precision_class = torch.exp(-self.log_vars[1])
                loss_class_weighted = precision_class * loss_class + self.log_vars[1]

                # Calculate ranking loss
                loss_rank_scalar = torch.mean(loss_rank)
                
                precision_rank = torch.exp(-self.log_vars[0])
                loss_rank_weighted = precision_rank * loss_rank_scalar + self.log_vars[0]
                
                # Calculate total loss with balance factor
                total_loss = (rank_class_balance * loss_rank_weighted + loss_class_weighted) / (rank_class_balance + 1)

                # Optimizer Step
                self.discriminator.opt_zero_grad()
                total_loss.backward()

                # Apply entity embedding constraints (e.g. norm <= 1)
                self.discriminator.opt_step()

                # Update Metrics
                epoch_reward += torch.sum(rewards)
                epoch_d_loss += total_loss.item() # Logging the combined loss

                # Update Generator
                rewards = rewards - avg_reward

                # Update generator with rewards
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
                rank_metrics = self.discriminator.evaluate_on_ranking(valid_data_w_label, heads, tails, filt=True, k_list=None)
                class_metrics = self.discriminator.evaluate_on_classification(valid_data_w_label, optimizing_metric='accuracy', threshold=None)
                perf = (rank_class_balance * rank_metrics['mrr'] + class_metrics['accuracy']) / (rank_class_balance + 1)

                logging.info('Validation at epoch %d: MRR=%f, Accuracy=%f, Perf=%f', 
                            epoch + 1, rank_metrics['mrr'], class_metrics['accuracy'], perf)
                if perf > best_perf:
                    print(f"Saving KBGAN at epoch {epoch + 1} with performance: {best_perf}")
                    best_perf = perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if use_early_stopping and patience_counter >= patience:
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, patience)
                    break

        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        if is_save_kbgan:
            print(f"Saving trained KBGAN (discriminator) with performance: {best_perf}")
            self.kbgan_path = self.save_kbgan()
            print(f"Saved trained KBGAN (discriminator) successfully to: {self.kbgan_path}")
            return best_perf, self.kbgan_path
        return best_perf, None

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                        filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if (not self.discriminator.is_trained_or_loaded()):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.evaluate_on_ranking(test_data, heads, tails, filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_triple_classification(self, test_data_with_labels: tuple, optimizing_metric: str='accuracy') -> Tuple[dict, float, list, list]:
        if (not self.discriminator.is_trained_or_loaded()):
            raise ValueError("KBGAN (discriminator) must be trained before being tested!")
        
        print("Evaluating KBGAN discriminator on Triple Classification...")
        threshold = (self.max_d_good + self.min_d_bad) / 2
        logging.info(f"Using threshold: {threshold:.4f} for triple classification based on observed max_d_good: {self.max_d_good:.4f} and min_d_bad: {self.min_d_bad:.4f}")
        metrics = self.discriminator.evaluate_on_classification(test_data_with_labels, optimizing_metric=optimizing_metric, threshold=threshold)
        return metrics