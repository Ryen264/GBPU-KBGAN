import os
import logging
import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
from typing import Generator, Tuple
import numpy as np

from datasets import batch_by_num, batch_by_size, convert_data_to_no_label, BernCorrupterMulti, BernCorrupter
from models import TransE, TransD, DistMult, ComplEx
import config
import metrics

EPSILON = 1e-30
FILTER_RANKING_PENALTY = 1e30
OPTIMIZER_MAP = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}

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
        if self.model_type == 'TransE':
            self.model = TransE(self.n_entity, self.n_relation)
        elif self.model_type == 'TransD':
            self.model = TransD(self.n_entity, self.n_relation)
        elif self.model_type == 'DistMult':
            self.model = DistMult(self.n_entity, self.n_relation)
        elif self.model_type == 'ComplEx':
            self.model = ComplEx(self.n_entity, self.n_relation)

        self.model_path = self.model.model_path
        print(f"Initialized component successfully: {self.model_type} model with role {self.role}, n_entity={self.n_entity}, n_relation={self.n_relation}.")

    def load(self, model_path: str) -> None:
        if (self.n_entity is None or self.n_relation is None):
            raise ValueError("Component must be fitted before being loaded!")

        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation)
        self.model.load(model_path)
        print(f"Loaded component successfully by: {model_path}")

    def save(self, model_path: str=None):
        if model_path is None:
            model_path = self.model.model_path

        self.model.save(model_path)
        print(f"Saved component successfully by: {model_path}")

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_var, relation_var, tail_var = Variable(head.to(config.device)), Variable(relation.to(config.device)), Variable(tail.to(config.device))
        return self.model.score(head_var, relation_var, tail_var)

    def train(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
              rank_class_balance: float = 1.0, early_stop_patience: int=-1,
              rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
              class_optimizing_metric: str='accuracy', class_threshold: float=None) -> float:    
        config.overwrite_config_with_args(["--log.prefix=" + self.model_type + '_'])
        config.logger_init()

        if self.model_type in ['TransE', 'TransD']:
            corrupter = BernCorrupter(train_data, self.n_entity, self.n_relation)
        elif self.model_type in ['DistMult', 'ComplEx']:
            corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, self.model.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")\
            
        valid_data_no_label = convert_data_to_no_label(valid_data_w_label)
        rank_metrics = lambda: self.evaluate_on_ranking(valid_data_no_label, heads, tails,
                                                        filt=rank_filt, k_list=rank_k_list)
        
        class_metrics = lambda: self.evaluate_on_classification(valid_data_w_label,
                                                                optimizing_metric=class_optimizing_metric, threshold=class_threshold)
        tester = lambda: (rank_class_balance * rank_metrics()[rank_optimizing_metric] + class_metrics()[class_optimizing_metric]) / (rank_class_balance + 1)

        best_perf = self.model.train(train_data,
                                     corrupter, tester, early_stop_patience=early_stop_patience)
        print(f'Trained component successfully: {self.model_type} model.')
        return best_perf
    
    def opt_zero_grad(self) -> None:
        self.model.ensure_optimizer()
        self.model.opt.zero_grad()

    def opt_step(self) -> None:
        self.model.opt.step()
        self.model.constraint()

    def set_optimizer(self, optimizer_name: str, lr: float, weight_decay: float=0.0) -> None:
        opt_cls = OPTIMIZER_MAP.get(optimizer_name, Adam)
        try:
            self.model.opt = opt_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        except (AttributeError, TypeError):
            pass

    def generator_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                        n_sample: int=1, temperature: float=1.0, train: bool=True) -> Generator[torch.Tensor, torch.Tensor, None]:
        """
        Generator step: sample fake triples and update with REINFORCE
        """
        if (self.role != "generator"):
            raise ValueError("This component is not a generator!")
        
        # Forward pass: generate samples
        n, m = tail.size()

        relation_var, head_var, tail_var = Variable(relation.to(config.device)), Variable(head.to(config.device)), Variable(tail.to(config.device))

        logits = self.model.prob_logit(head_var, relation_var, tail_var) / temperature
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
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.to(config.device), sample_idx.data])
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

        # Forward pass: compute losses and scores
        head_var, relation_var, tail_var = Variable(head.to(config.device)), Variable(relation.to(config.device)), Variable(tail.to(config.device))        
        head_fake_var, tail_fake_var = Variable(head_fake.to(config.device)), Variable(tail_fake.to(config.device))

        if head_fake_var.dim() == relation_var.dim() + 1:
            relation_fake_var = relation_var.unsqueeze(1).expand_as(head_fake_var)
            d_good = self.model.dist(head_var, relation_var, tail_var).unsqueeze(1).expand_as(head_fake_var)
        else:
            relation_fake_var = relation_var
            d_good = self.model.dist(head_var, relation_var, tail_var)

        d_bad = self.model.dist(head_fake_var, relation_fake_var, tail_fake_var)
        pair_loss = nnf.relu(d_good - d_bad + self.model.margin)
        fake_scores = self.model.score(head_fake_var, relation_fake_var, tail_fake_var)
                
        # Backward pass: update discriminator
        if train:
            self.opt_zero_grad()
            sum_loss = torch.sum(pair_loss)
            sum_loss.backward()
            self.opt_step()
        # In training mode, return differentiable pair_loss so caller can build a joint objective.
        # In evaluation mode, detach to avoid building autograd graph.
        pair_loss_out = pair_loss if train else pair_loss.detach()
        return pair_loss_out, -fake_scores.detach(), d_good.detach().max().item(), d_bad.detach().min().item()

    def evaluate_on_ranking(self, test_data: tuple, heads: torch.Tensor, tails: torch.Tensor,
                            filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        count = 0
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(self.model.test_batch_size, *test_data):
                # Convert lists to tensors
                batch_head = torch.LongTensor(batch_head)
                batch_relation = torch.LongTensor(batch_relation)
                batch_tail = torch.LongTensor(batch_tail)
                batch_size = batch_head.size(0)

                all_var = torch.arange(0, self.n_entity).unsqueeze(0).expand(batch_size, self.n_entity).long().to(config.device)
                head_var = batch_head.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)

                batch_head_scores = self.model.score(all_var, relation_var, tail_var)
                batch_tail_scores = self.model.score(head_var, relation_var, all_var)
            
                batch_head_scores = batch_head_scores.detach()
                batch_tail_scores = batch_tail_scores.detach()

                for head, relation, tail, head_scores, tail_scores in zip(batch_head, batch_relation, batch_tail, batch_head_scores, batch_tail_scores):
                    head_id, relation_id, tail_id = head.item(), relation.item(), tail.item()
                    if filt:
                        key_head = (tail_id, relation_id)
                        if key_head in heads and heads[key_head]._nnz() > 1:
                            tmp = head_scores[head_id].item()
                            head_scores += heads[key_head].to(config.device) * FILTER_RANKING_PENALTY
                            head_scores[head_id] = tmp
                            
                        key_tail = (head_id, relation_id)
                        if key_tail in tails and tails[key_tail]._nnz() > 1:
                            tmp = tail_scores[tail_id].item()
                            tail_scores += tails[key_tail].to(config.device) * FILTER_RANKING_PENALTY
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

    def evaluate_on_classification(self, test_data_w_label: tuple,
                                   optimizing_metric: str='accuracy', threshold: float=None) -> dict:       
        def find_optimal_threshold(valid_data: tuple, labels: list, n_thresholds: int=100) -> Tuple[float, bool]:
            """
            Find the optimal threshold for triple classification using validation data.

            Args:
                valid_data: Tuple of (heads, relations, tails)
                labels: Ground truth labels for validation data
                n_thresholds: Number of threshold values to try

            Returns:
                (Optimal threshold value, positive_if_lower)
                where positive_if_lower=True means score < threshold predicts positive.
            """
            heads, relations, tails = valid_data

            # Compute scores for all validation samples (batched for efficiency)
            scores_list = []
            with torch.no_grad():
                for batch_head, batch_relation, batch_tail in batch_by_size(self.model.test_batch_size,
                                                                           heads, relations, tails):
                    head_var = torch.LongTensor(batch_head).to(config.device)
                    relation_var = torch.LongTensor(batch_relation).to(config.device)
                    tail_var = torch.LongTensor(batch_tail).to(config.device)

                    batch_scores = self.model.score(head_var, relation_var, tail_var)
                    batch_scores = batch_scores.detach().cpu().numpy()
                    scores_list.extend(batch_scores.tolist())

            # Try different threshold values
            scores_array = np.array(scores_list)
            min_score = float(scores_array.min())
            max_score = float(scores_array.max())
            threshold_values = np.linspace(min_score, max_score, n_thresholds)

            best_val = -float('inf')
            best_threshold = 0.0
            best_positive_if_lower = self.model.is_distance_based

            for threshold in threshold_values:
                predictions_lower = np.where(scores_array < threshold, 1, 0).tolist()
                metrics_lower = metrics.classification_metrics(predictions_lower, labels, scores=scores_list)
                test_val_lower = metrics_lower.get(optimizing_metric, 0.0)

                predictions_higher = np.where(scores_array > threshold, 1, 0).tolist()
                metrics_higher = metrics.classification_metrics(predictions_higher, labels, scores=scores_list)
                test_val_higher = metrics_higher.get(optimizing_metric, 0.0)

                if test_val_lower >= test_val_higher:
                    candidate_val = test_val_lower
                    candidate_positive_if_lower = True
                else:
                    candidate_val = test_val_higher
                    candidate_positive_if_lower = False

                if candidate_val > best_val:
                    best_val = candidate_val
                    best_threshold = threshold
                    best_positive_if_lower = candidate_positive_if_lower

            direction_str = '<' if best_positive_if_lower else '>'
            logging.info(f"Optimal threshold: score {direction_str} {best_threshold:.4f} => positive ({optimizing_metric}={best_val:.4f})")
            return best_threshold, best_positive_if_lower

        if len(test_data_w_label) < 4:
            raise ValueError("For classification metrics, test_data_w_label must include labels as the 4th element (heads, relations, tails, labels).")
        
        heads_list, relations_list, tails_list, labels = test_data_w_label
        scores_list = []
        true_labels = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(self.model.test_batch_size,
                                                                                     heads_list, relations_list, tails_list, labels):
                # ensure tensors on device
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)

                batch_scores = self.model.score(head_var, relation_var, tail_var)
                batch_scores = batch_scores.detach().cpu().tolist()

                scores_list.extend([float(s) for s in batch_scores])
                true_labels.extend([int(x) for x in batch_label])

        if len(scores_list) == 0:
            raise ValueError("No samples found in test_data for classification evaluation.")

        if threshold is None:
            n_thresholds = 100
            threshold, positive_if_lower = find_optimal_threshold(valid_data=(heads_list, relations_list, tails_list), labels=true_labels, n_thresholds=n_thresholds)
            direction_str = '<' if positive_if_lower else '>'
            logging.info(f"Determined optimal threshold for classification: score {direction_str} {threshold:.4f} => positive, using validation data with {n_thresholds} thresholds.")
        else:
            positive_if_lower = self.model.is_distance_based

        # Vectorized prediction generation
        scores_array = np.array(scores_list)
        if positive_if_lower:
            predictions = np.where(scores_array < threshold, 1, 0).tolist()
        else:
            predictions = np.where(scores_array > threshold, 1, 0).tolist()

        # Convert scores so that larger values indicate positive class for AUC
        scores_for_auc = [-s for s in scores_list] if positive_if_lower else scores_list

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
        self.generator_type = generator_type
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.discriminator = Component(role="discriminator", model_type=discriminator_type,
                                       n_entity=n_entity, n_relation=n_relation)
        self.generator = Component(role="generator", model_type=generator_type,
                                   n_entity=n_entity, n_relation=n_relation)

        self.discriminator_path = self.discriminator.model_path
        self.generator_path = self.generator.model_path

        self.dataset = config._config.dataset
        self.task = config._config.task
        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
        os.makedirs(self.task_dir, exist_ok=True)

        self.model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        self.kbgan_path = os.path.join(self.task_dir, self.model_name)
        self.max_d_good = 0.0
        self.min_d_bad = float('inf')
        self.optimal_threshold = None

    def load_discriminator(self, filepath: str) -> None:
        self.discriminator.load(filepath)
        print(f"Loaded discriminator successfully by path: {filepath}")

    def load_generator(self, filepath: str) -> None:
        self.generator.load(filepath)
        print(f"Loaded generator successfully by path: {filepath}")

    def load_kbgan(self, filepath: str) -> None:      
        self.discriminator.load(filepath)
        print(f"Loaded KBGAN (discriminator) successfully by: {filepath}")

    def save_kbgan(self, filepath: str=None) -> None:
        if filepath is None:
            filepath = self.kbgan_path
        self.discriminator.save(filepath)
        print(f"Saved KBGAN (discriminator) successfully to: {filepath}")
        
    def train_components(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                        rank_class_balance: float=5.0, early_stop_patience: int=-1,
                        rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
                        class_optimizing_metric: str='accuracy', class_threshold: float=None) -> Tuple[float, float]:
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        best_perf_d = self.discriminator.train(heads, tails, train_data, valid_data_w_label,
                                                rank_class_balance=rank_class_balance, early_stop_patience=early_stop_patience,
                                                rank_optimizing_metric=rank_optimizing_metric, rank_filt=rank_filt, rank_k_list=rank_k_list,
                                                class_optimizing_metric=class_optimizing_metric, class_threshold=class_threshold)
        print(f"Trained {self.discriminator_type} discriminator successfully with performance: {best_perf_d}")

        best_perf_g = self.generator.train(heads, tails, train_data, valid_data_w_label,
                                            rank_class_balance=rank_class_balance, early_stop_patience=early_stop_patience,
                                            rank_optimizing_metric=rank_optimizing_metric, rank_filt=rank_filt, rank_k_list=rank_k_list,
                                            class_optimizing_metric=class_optimizing_metric, class_threshold=class_threshold)
        print(f"Trained {self.generator_type} generator successfully with performance: {best_perf_g}")
        return best_perf_d, best_perf_g
           
    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                class_rank_balance: float=1.0, early_stop_patience: int=-1,
                temperature: float=1.0, n_sample: int=20, n_epoch: int=5000, n_batch: int=100, epoch_per_test: int=100,
                rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
                class_optimizing_metric: str='accuracy', class_use_maxgood_minbad_threshold: bool=True) -> float:
        """
        If class_rank_balance == 0, only optimize ranking loss (only for Link Prediction).
        """
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        # Define Classification Loss function
        bce_criterion = torch.nn.BCELoss()

        corrupter = BernCorrupterMulti(train_data, self.n_entity, self.n_relation, n_sample)
        head, relation, tail = train_data
        n_train = len(head)

        best_perf = 0.0
        avg_reward = 0.0
        patience_counter = 0
        for epoch in range(n_epoch):
            epoch_d_loss = 0
            epoch_rank_loss = 0
            epoch_class_loss = 0
            epoch_reward = 0

            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(n_batch, head, relation, tail, head_cand, relation_cand, tail_cand, n_sample=n_train):             
                # --- Generator Step ---
                gen_step = self.generator.generator_step(hs, rs, ts,
                                                        n_sample=n_sample, temperature=temperature,
                                                        train=True)
                head_smpl, tail_smpl = next(gen_step)
                head_smpl_device, tail_smpl_device = head_smpl.to(config.device), tail_smpl.to(config.device)
                
                # --- Discriminator Step ---
                # 1. Get Ranking Loss (and rewards for Generator)
                loss_rank, rewards, max_d_good, min_d_bad = self.discriminator.discriminator_step(head=h, relation=r, tail=t,
                                                                                                  head_fake=head_smpl_device, tail_fake=tail_smpl_device,
                                                                                                  train=True)

                # Update max_d_good and min_d_bad for monitoring
                update_threshold = False
                if max_d_good > self.max_d_good:
                    logging.info(f"Updated max_d_good: {self.max_d_good:.4f} -> {max_d_good:.4f}")
                    self.max_d_good = max_d_good
                    update_threshold = True
                if min_d_bad < self.min_d_bad:
                    logging.info(f"Updated min_d_bad: {self.min_d_bad:.4f} -> {min_d_bad:.4f}")
                    self.min_d_bad = min_d_bad
                    update_threshold = True

                # Calculate optimal threshold for classification based on observed max_d_good and min_d_bad
                if update_threshold:
                    self.optimal_threshold = (self.max_d_good + self.min_d_bad) / 2
                    logging.info(f"Updated optimal_threshold: {self.optimal_threshold:.4f} based on max_d_good and min_d_bad.")                    

                # 2. Get Classification Loss (BCE)
                # We need raw scores from the discriminator to compute BCE.
                pos_score = self.discriminator.score(h, r, t)
                if head_smpl_device.dim() == r.dim() + 1:
                    relation_for_neg = r.unsqueeze(1).expand_as(head_smpl_device)
                else:
                    relation_for_neg = r
                neg_score = self.discriminator.score(head_smpl_device, relation_for_neg, tail_smpl_device)
                
                # Normalize scores to [0, 1] using sigmoid and clamp to avoid numerical issues
                pos_score_norm = torch.clamp(torch.sigmoid(pos_score), min=EPSILON, max=1.0-EPSILON).float()
                neg_score_norm = torch.clamp(torch.sigmoid(neg_score), min=EPSILON, max=1.0-EPSILON).float()
                
                # Target: 1 for Real, 0 for Fake
                target_pos = torch.ones_like(pos_score_norm, dtype=torch.float32)
                target_neg = torch.zeros_like(neg_score_norm, dtype=torch.float32)

                # Calculate classification loss
                loss_class = bce_criterion(pos_score_norm, target_pos) + bce_criterion(neg_score_norm, target_neg)

                # Calculate ranking loss
                loss_rank_scalar = torch.mean(loss_rank)

                # Joint objective using only rank_class_balance
                total_loss = (loss_rank_scalar + class_rank_balance * loss_class) / (1 + class_rank_balance)

                # Optimizer Step
                self.discriminator.opt_zero_grad()
                total_loss.backward()

                # Apply entity embedding constraints (e.g. norm <= 1)
                self.discriminator.opt_step()

                # Update Metrics
                epoch_reward += torch.sum(rewards)
                epoch_d_loss += total_loss.item() # Logging the combined loss
                epoch_rank_loss += loss_rank_scalar.item()
                epoch_class_loss += loss_class.item()

                # Update Generator
                rewards = rewards - avg_reward

                # Update generator with rewards
                try:
                    rewards_for_gen = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
                    gen_step.send(rewards_for_gen)
                except StopIteration:
                    pass
                
            avg_loss = epoch_d_loss / n_train
            avg_rank_loss = epoch_rank_loss / n_train
            avg_class_loss = epoch_class_loss / n_train
            avg_reward = epoch_reward / n_train

            logging.info(f'Epoch {epoch + 1}/{n_epoch}, Joint_Loss={avg_loss}, Rank_Loss={avg_rank_loss}, Class_Loss={avg_class_loss}')

            if (epoch + 1) % epoch_per_test == 0:
                valid_data_no_label = valid_data_w_label[:3]
                rank_metrics = self.discriminator.evaluate_on_ranking(valid_data_no_label, heads, tails,
                                                                      filt=rank_filt, k_list=rank_k_list)
                
                class_threshold = self.optimal_threshold if class_use_maxgood_minbad_threshold else None
                class_metrics = self.discriminator.evaluate_on_classification(valid_data_w_label,
                                                                              optimizing_metric=class_optimizing_metric, threshold=class_threshold)
                
                test_perf = (class_rank_balance * rank_metrics[rank_optimizing_metric] + class_metrics[class_optimizing_metric]) / (class_rank_balance + 1)
                logging.info(f'Validation at epoch {epoch + 1}: {rank_optimizing_metric}={rank_metrics[rank_optimizing_metric]}, {class_optimizing_metric}={class_metrics[class_optimizing_metric]}, Perf={test_perf}')
                
                if test_perf > best_perf:
                    self.save_kbgan()  # Save the best model
                    print(f"Saved KBGAN at epoch {epoch + 1} with performance: {best_perf}")
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    logging.info(f'Early stopping triggered at epoch {epoch + 1} (patience={early_stop_patience})')
                    break
        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        return best_perf

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                    filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.evaluate_on_ranking(test_data, heads, tails,
                                                        filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_triple_classification(self, test_data_w_label: tuple,
                                          optimizing_metric: str='accuracy', use_maxgood_minbad_threshold: bool=True) -> dict:
        if not isinstance(test_data_w_label[0], torch.Tensor):
            test_data_w_label = [torch.LongTensor(vec) for vec in test_data_w_label]
            
        print("Evaluating KBGAN discriminator on Triple Classification...")
        threshold = self.optimal_threshold if use_maxgood_minbad_threshold else None
        metrics = self.discriminator.evaluate_on_classification(test_data_w_label,
                                                                optimizing_metric=optimizing_metric, threshold=threshold)
        return metrics