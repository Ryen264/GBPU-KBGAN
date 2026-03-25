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
        self.classification_threshold = None
        self.best_threshold_perf = {}
        print(f"Initialized component successfully: {self.model_type} model with role {self.role}, n_entity={self.n_entity}, n_relation={self.n_relation}.")

        self.class_threshold = None

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
            class_rank_balance: float = 1.0, early_stop_patience: int=-1,
            rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
            class_optimizing_metric: str='accuracy') -> float:    
        config.overwrite_config_with_args(["--log.prefix=" + self.model_type + '_'])
        config.logger_init()

        if class_rank_balance < 0.0 or class_rank_balance > 1.0:
            raise ValueError("class_rank_balance must be in [0, 1].")

        # Convert to plain lists for BernCorrupter so dict keys use value-based hashing
        train_data_list = [d.tolist() if isinstance(d, torch.Tensor) else d for d in train_data]
        if self.model_type in ['TransE', 'TransD']:
            corrupter = BernCorrupter(train_data_list, self.n_entity, self.n_relation)
        elif self.model_type in ['DistMult', 'ComplEx']:
            corrupter = BernCorrupterMulti(train_data_list, self.n_entity, self.n_relation, self.model.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")\
            
        valid_data_no_label = convert_data_to_no_label(valid_data_w_label)
        rank_metrics = lambda: self.evaluate_on_ranking(valid_data_no_label, heads, tails,
                                                        filt=rank_filt, k_list=rank_k_list)
        
        class_metrics = lambda: self.evaluate_on_classification(valid_data_w_label,
                                                                optimizing_metric=class_optimizing_metric, is_threshold_tunning=True)
        tester = lambda: ((1.0 - class_rank_balance) * rank_metrics()[rank_optimizing_metric]) + (class_rank_balance * class_metrics()[class_optimizing_metric])

        print(f"Start training component: {self.model_type} model with role {self.role}...")
        best_perf, best_epoch = self.model.train(train_data,
                                                corrupter, tester, early_stop_patience=early_stop_patience)
        print(f'Trained component successfully: {self.model_type} model.')
        return best_perf, best_epoch
    
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
                    n_sample: int=1, temperature: float=1.0, train: bool=True,
                    sampling_strategy: str='multinomial') -> Generator[torch.Tensor, torch.Tensor, None]:
        """
        Generator step: sample fake triples and update with REINFORCE.
        n_sample controls number of sampled fake triples per input triple.
        sampling_strategy: 'multinomial' or 'topk'.
        """
        if (self.role != "generator"):
            raise ValueError("This component is not a generator!")
        if sampling_strategy not in ['multinomial', 'topk']:
            raise ValueError("sampling_strategy must be one of ['multinomial', 'topk']")
        
        # Forward pass: generate samples
        n, m = tail.size()
        if n_sample > m:
            raise ValueError(f"n_sample ({n_sample}) cannot be larger than candidate pool size ({m}).")

        relation_var, head_var, tail_var = Variable(relation.to(config.device)), Variable(head.to(config.device)), Variable(tail.to(config.device))

        logits = self.model.prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits, dim=-1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        if sampling_strategy == 'topk':
            sample_idx = torch.topk(probs, k=n_sample, dim=-1, largest=True, sorted=False).indices
        else:
            sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]
        
        # Yield samples to get rewards from discriminator
        rewards = yield sample_heads, sample_tails
        
        # Backward pass: update generator with REINFORCE
        if train:
            self.opt_zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            # Move indices to device for proper indexing
            row_idx_device = row_idx.to(config.device)
            sample_idx_device = sample_idx.to(config.device)
            # Ensure rewards are a tensor and on device
            rewards_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, device=config.device, dtype=torch.float32)
            if rewards_tensor.device != config.device:
                rewards_tensor = rewards_tensor.to(config.device)
            # Compute REINFORCE loss: -sum(rewards * log_probs) 
            reinforce_loss = -torch.sum(rewards_tensor * log_probs[row_idx_device, sample_idx_device])
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
        
        # Format metrics for cleaner output
        parts = []
        label_map = {'mr': 'MR', 'mrr': 'MRR'}
        for k, v in ranking_metrics.items():
            label = label_map.get(k, k.replace('hit@', 'Hit@'))
            parts.append(f"{label}: {v:.4f}")
        ranking_metrics_str = f"Ranking metrics: {', '.join(parts)}\n"
        logging.info(ranking_metrics_str)
        return ranking_metrics
    
    def find_optimal_threshold(self, valid_data: tuple, labels: list,
                            n_thresholds: int=100, optimizing_metric: str='accuracy') -> Tuple[float, float]:
            """
            Find the optimal threshold for triple classification using validation data.

            Args:
                valid_data: Tuple of (heads, relations, tails)
                labels: Ground truth labels for validation data
                n_thresholds: Number of threshold values to try

            Returns:
                (Optimal threshold value, best validation score for optimizing_metric)
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

            for threshold in threshold_values:
                predictions = np.where(scores_array < threshold, 1, 0).tolist()
                scores_for_auc = (-scores_array).tolist()
                candidate_metrics = metrics.classification_metrics(predictions, labels, scores=scores_for_auc)
                candidate_val = candidate_metrics.get(optimizing_metric, 0.0)

                if candidate_val > best_val:
                    best_val = candidate_val
                    best_threshold = threshold
            return best_threshold, best_val
    
    def evaluate_on_classification(self, test_data_w_label: tuple, optimizing_metric: str='accuracy',
                                is_threshold_tunning: bool=False, external_threshold: float = None) -> dict:       
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
        threshold = None
        if is_threshold_tunning:
            n_thresholds = 100
            candidate_threshold, candidate_val = self.find_optimal_threshold(valid_data=(heads_list, relations_list, tails_list),
                                                                            labels=true_labels, n_thresholds=n_thresholds,
                                                                            optimizing_metric=optimizing_metric)
            best_so_far = self.best_threshold_perf.get(optimizing_metric, -float('inf'))
            if self.classification_threshold is None or candidate_val > best_so_far:
                self.classification_threshold = candidate_threshold
                self.best_threshold_perf[optimizing_metric] = candidate_val
                logging.info(
                    f"Updated best classification threshold ({optimizing_metric}={candidate_val:.4f}): "
                    f"score < {self.classification_threshold:.4f} => positive."
                )
            else:
                logging.info(
                    f"Kept previous best classification threshold ({optimizing_metric}={best_so_far:.4f}). "
                    f"Current tuned threshold score is {candidate_val:.4f}."
                )
            threshold = self.classification_threshold
            logging.info(
                f"Using classification threshold: score < {self.classification_threshold:.4f} => positive "
                f"(searched {n_thresholds} thresholds)."
            )
        else:
            if external_threshold is not None:
                threshold = external_threshold
            else:
                threshold = self.classification_threshold

        # Fixed rule in this codebase: lower score means more likely positive.
        scores_array = np.array(scores_list)
        predictions = np.where(scores_array < threshold, 1, 0).tolist()

        # Convert scores so that larger values indicate positive class for AUC
        scores_for_auc = [-s for s in scores_list]

        classification_metrics = metrics.classification_metrics(predictions, true_labels, scores=scores_for_auc)
        # Format metrics for cleaner output
        parts = []
        label_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'pr_auc': 'PR AUC', 'roc_auc': 'ROC AUC'}
        for k, v in classification_metrics.items():
            label = label_map.get(k, k)
            parts.append(f"{label}: {v:.4f}")
        classification_metrics_str = f"Classification metrics: {', '.join(parts)}\n"

        threshold_source = None
        if external_threshold is not None:
            threshold_source = "External threshold"
        else:
            threshold_source = "Internal threshold"
        
        logging.info(classification_metrics_str)
        logging.info(
            f"[Classification] threshold_source={threshold_source}, "
            f"threshold={threshold:.6f}, decision_rule='score<threshold=>positive'"
        )
        
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
                        class_rank_balance: float=1.0, early_stop_patience: int=-1,
                        rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
                        class_optimizing_metric: str='accuracy') -> Tuple[float, float]:
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        best_perf_d, best_epoch_d = self.discriminator.train(heads, tails, train_data, valid_data_w_label,
                                                            class_rank_balance=class_rank_balance,
                                                            early_stop_patience=early_stop_patience,
                                                            rank_optimizing_metric=rank_optimizing_metric,
                                                            rank_filt=rank_filt, rank_k_list=rank_k_list,
                                                            class_optimizing_metric=class_optimizing_metric)
        print(f"Trained {self.discriminator_type} discriminator successfully with performance: {best_perf_d}, epoch: {best_epoch_d}")

        best_perf_g, best_epoch_g = self.generator.train(heads, tails, train_data, valid_data_w_label,
                                                        class_rank_balance=class_rank_balance,
                                                        early_stop_patience=early_stop_patience,
                                                        rank_optimizing_metric=rank_optimizing_metric,
                                                        rank_filt=rank_filt, rank_k_list=rank_k_list,
                                                        class_optimizing_metric=class_optimizing_metric)
        print(f"Trained {self.generator_type} generator successfully with performance: {best_perf_g}, epoch: {best_epoch_g}")
        return best_perf_d, best_perf_g
           
    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                class_rank_balance: float=1.0, early_stop_patience: int=-1,
                temperature: float=1.0, n_sample: int=20, n_candidate: int=None,
                n_generated_valid_negative: int=0,
                n_epoch: int=5000, n_batch: int=100, epoch_per_test: int=100,
                rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
                class_optimizing_metric: str='accuracy', class_use_maxgood_minbad_threshold: bool=True,
                class_true_percentile: float=95.0, class_fake_percentile: float=5.0,
                class_true_fake_balance: float=0.33,
                class_rank_balance_start: float=None, class_rank_balance_warmup_epochs: int=0,
                negative_sampling_strategy: str='multinomial',
                join_loss_method: str='adaptive_weight', loss_ema_beta: float=0.98) -> float:
        """
        class_rank_balance is a ratio in [0, 1]:
        - 0.0 => optimize ranking only
        - 1.0 => optimize classification only

        class_rank_balance_start and class_rank_balance_warmup_epochs optionally define
        a linear optimization schedule from start -> class_rank_balance.

        joint_loss_method:
        - 'fixed': total = (1-b)*rank + b*class
        - 'adaptive_norm': normalize each loss by EMA magnitude before mixing
        - 'adaptive_weight': apply EMA inverse-loss weighting on raw losses before mixing

        negative_sampling_strategy:
        - 'multinomial': sample negatives from generator distribution
        - 'topk': choose highest-probability negatives (harder)
        """
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        # Convert to plain lists for BernCorrupterMulti so dict keys use value-based hashing
        train_data_list = [d.tolist() if isinstance(d, torch.Tensor) else d for d in train_data]

        # [ORIGINAL KBGAN]
        corrupter = BernCorrupterMulti(train_data_list, self.n_entity, self.n_relation, n_candidate)
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        avg_reward = 0.0

        # [CLASS TASK]
        #   Define classification loss in logit space for numerical stability.
        #   Use -score as positive-class logit so lower score => more likely positive.
        do_class_task = (class_rank_balance > 0.0)
        if do_class_task:
            bce_logits_criterion = torch.nn.BCEWithLogitsLoss()

        # [EARLY STOPPING]
        patience_counter = 0

        # [EMA BETA]
        ema_rank = None
        ema_class = None

        # [WARM-UP BALANCE]
        warmup_complete = False

        for epoch in range(n_epoch):
            # [WARM-UP BALANCE]
            #   Compute class_rank_balance_opt: linear warmup from start -> target over warmup_epochs
            if class_rank_balance_warmup_epochs > 0 and not warmup_complete:
                if epoch + 1 >= class_rank_balance_warmup_epochs:
                    # Warmup complete: use target balance
                    class_rank_balance_opt = class_rank_balance
                    warmup_complete = True
                else:
                    # Still in warmup: interpolate linearly
                    warmup_progress = float(epoch + 1) / float(class_rank_balance_warmup_epochs)
                    class_rank_balance_opt = class_rank_balance_start + \
                        (class_rank_balance - class_rank_balance_start) * warmup_progress
            else:
                class_rank_balance_opt = class_rank_balance

            # [ORIGINAL KBGAN]
            epoch_d_loss = 0
            epoch_reward = 0.0

            # [JOINT LOSS]
            epoch_rank_loss = 0
            epoch_class_loss = 0

            # [ORIGINAL KBGAN]
            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(n_batch, head, relation, tail,
                                                    head_cand, relation_cand, tail_cand, n_sample=n_train):             
                # [JOINT LOSS]
                batch_size = h.size(0)

                # --- Generator Step ---
                # [ORIGINAL KBGAN]
                gen_step = self.generator.generator_step(hs, rs, ts, n_sample=n_sample, temperature=temperature,
                                                        train=True, sampling_strategy=negative_sampling_strategy)
                head_smpl, tail_smpl = next(gen_step)

                # [GPU]
                head_smpl_device, tail_smpl_device = head_smpl.to(config.device), tail_smpl.to(config.device)
                
                # --- Discriminator Step ---
                # 1. Get Ranking Loss (and rewards for Generator)
                # [ORIGINAL KBGAN]
                rank_loss, rewards, d_good_max, d_bad_min = self.discriminator.discriminator_step(head=h, relation=r, tail=t,
                                                                                head_fake=head_smpl_device, tail_fake=tail_smpl_device,
                                                                                train=True)
                #   Update Metrics
                epoch_reward += float(torch.sum(rewards).item())

                #   Update Generator
                rewards = rewards - avg_reward

                #   Update generator with rewards
                rewards_for_gen = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
                gen_step.send(rewards_for_gen)

                # [RANK TASK]
                #   Calculate ranking loss
                rank_loss_scalar = torch.mean(rank_loss)

                # [CLASS TASK]
                if do_class_task:
                    # 2. Get Classification Loss (direction-consistent with score<threshold=>positive).
                    # Positive-class probability is modeled as sigmoid(-score).
                    pos_score = self.discriminator.score(h, r, t)

                    if head_smpl_device.dim() == r.dim() + 1:
                        relation_for_neg = r.unsqueeze(1).expand_as(head_smpl_device)
                    else:
                        relation_for_neg = r
                    neg_score = self.discriminator.score(head_smpl_device, relation_for_neg, tail_smpl_device)

                    pos_logits = -pos_score
                    neg_logits = -neg_score

                    # Target: 1 for Real, 0 for Fake
                    target_pos = torch.ones_like(pos_logits, dtype=torch.float32)
                    target_neg = torch.zeros_like(neg_logits, dtype=torch.float32)

                    # Calculate classification loss
                    class_loss_scalar = bce_logits_criterion(pos_logits, target_pos) + bce_logits_criterion(neg_logits, target_neg)

                    # [JOINT LOSS]
                    if join_loss_method in ['adaptive_norm', 'adaptive_weight']:
                        rank_val = float(rank_loss_scalar.detach().item())
                        class_val = float(class_loss_scalar.detach().item())

                        # [EMA BETA]
                        ema_rank = rank_val if ema_rank is None else (loss_ema_beta * ema_rank + (1.0 - loss_ema_beta) * rank_val)
                        ema_class = class_val if ema_class is None else (loss_ema_beta * ema_class + (1.0 - loss_ema_beta) * class_val)
                    # [JOINT LOSS]
                    if join_loss_method == 'adaptive_norm':
                        rank_loss_scaled = rank_loss_scalar / (ema_rank + EPSILON)
                        class_loss_scaled = class_loss_scalar / (ema_class + EPSILON)

                        rank_weight = (1.0 - class_rank_balance_opt)
                        class_weight = class_rank_balance_opt                        
                    elif join_loss_method == 'adaptive_weight':
                        rank_loss_scaled = rank_loss_scalar
                        class_loss_scaled = class_loss_scalar

                        inv_rank = (1.0 - class_rank_balance_opt) / (ema_rank + EPSILON)
                        inv_class = class_rank_balance_opt / (ema_class + EPSILON)
                        weight_sum = inv_rank + inv_class + EPSILON
                        rank_weight = inv_rank / weight_sum
                        class_weight = inv_class / weight_sum
                    else:
                        rank_loss_scaled = rank_loss_scalar
                        class_loss_scaled = class_loss_scalar

                        rank_weight = (1.0 - class_rank_balance_opt)
                        class_weight = class_rank_balance_opt
                    total_loss = (rank_weight * rank_loss_scaled) + (class_weight * class_loss_scaled)
                else:
                    # Only ranking loss (Link Prediction only)
                    class_loss_scalar = torch.tensor(0.0)

                    rank_loss_scaled = rank_loss_scalar
                    class_loss_scaled = class_loss_scalar

                    rank_weight = 1.0
                    class_weight = 0.0

                    total_loss = rank_loss_scalar
                # Optimizer Step
                self.discriminator.opt_zero_grad()
                total_loss.backward()

                # Apply entity embedding constraints (e.g. norm <= 1)
                self.discriminator.opt_step()

                # [ORIGINAL KBGAN]
                #   total_loss/rank_loss_scalar/class_loss are batch means, so weight by batch size
                #   then divide by n_train at epoch end.
                epoch_d_loss += total_loss.item() * batch_size

                # [JOINT LOSS]
                epoch_rank_loss += rank_loss_scalar.item() * batch_size
                epoch_class_loss += class_loss_scalar.item() * batch_size
            # [ORIGINAL KBGAN]       
            avg_loss = epoch_d_loss / n_train
            avg_reward = epoch_reward / n_train

            # [JOINT LOSS]
            avg_rank_loss = epoch_rank_loss / n_train
            avg_class_loss = epoch_class_loss / n_train

            # [ORIGINAL KBGAN]
            log_msg = f"Train epoch {epoch + 1}/{n_epoch}, D_loss={avg_loss:.6f}, reward={avg_reward:.6f}"

            # [JOINT LOSS]
            log_msg += f"\n\t\tRank_Loss={avg_rank_loss:.6f}, Class_Loss={avg_class_loss:.6f}"
            logging.info(log_msg)

            if (epoch + 1) % epoch_per_test == 0:
                # [RANK TASK]
                valid_data_no_label = valid_data_w_label[:3]
                rank_metrics = self.discriminator.evaluate_on_ranking(valid_data_no_label, heads, tails,
                                                                      filt=rank_filt, k_list=rank_k_list)
                # [CLASS TASK]
                if do_class_task:
                    class_threshold = None
                    is_threshold_tunning = False

                    # [MAXGOOD MINBAD THRESHOLD]
                    if class_use_maxgood_minbad_threshold:
                        class_threshold = self._compute_midpoint_threshold_from_labeled_data(
                            valid_data_w_label,
                            n_generated_valid_negative=n_generated_valid_negative,
                            temperature=temperature,
                            true_percentile=class_true_percentile,
                            fake_percentile=class_fake_percentile,
                            true_fake_balance=class_true_fake_balance
                        )

                        if class_threshold is not None:
                            self.optimal_threshold = class_threshold
                            logging.info(f"Using validation midpoint threshold: {class_threshold:.6f}")
                        else:
                            # Fallback: tune threshold from validation metrics when midpoint cannot be computed.
                            is_threshold_tunning = True
                            logging.warning("Validation midpoint threshold unavailable. Falling back to threshold tuning on validation set.")
                    else:
                        # Default behavior when midpoint rule is disabled: tune threshold on validation set.
                        is_threshold_tunning = True

                    class_metrics = self.discriminator.evaluate_on_classification(valid_data_w_label,
                                                                                  optimizing_metric=class_optimizing_metric,
                                                                                  is_threshold_tunning=is_threshold_tunning,
                                                                                  external_threshold=class_threshold)
                    
                    # [JOINT LOSS]
                    test_perf = ((1.0 - class_rank_balance) * rank_metrics[rank_optimizing_metric]) + (class_rank_balance * class_metrics[class_optimizing_metric])
                    log_msg = f"Valid epoch {epoch + 1}/{n_epoch}, perf={test_perf}"

                    # [JOINT LOSS]
                    log_msg += f"\n\t{rank_optimizing_metric}={rank_metrics[rank_optimizing_metric]}, {class_optimizing_metric}={class_metrics[class_optimizing_metric]}"
                else:
                    test_perf = rank_metrics[rank_optimizing_metric]
                    log_msg = f"Valid epoch {epoch + 1}/{n_epoch}, perf={test_perf}"
                logging.info(log_msg)

                # [ORIGINAL KBGAN]
                if test_perf > best_perf:
                    best_perf = test_perf
                    self.save_kbgan()
                    print(f"Saved KBGAN at epoch {epoch + 1} with performance: {best_perf}")

                    # [EARLY STOPPING]
                    patience_counter = 0
                else:
                    # [EARLY STOPPING]
                    patience_counter += 1
                # [EARLY STOPPING]
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1} (patience={early_stop_patience})")
                    break
        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        return best_perf

    def _compute_midpoint_threshold_from_labeled_data(self, data_w_label: tuple,
                                                    n_generated_valid_negative: int=0, temperature: float=1.0,
                                                    true_percentile: float=95.0, fake_percentile: float=5.0,
                                                    true_fake_balance: float=0.33) -> float:
        """
        Compute midpoint threshold from labeled triples (and optional generator negatives)
        using percentile statistics:
        threshold = (percentile(true scores, true_percentile)
                     + percentile(fake scores, fake_percentile)) / 2.
        """
        if len(data_w_label) < 4:
            return None
        
        if not (0.0 <= true_percentile <= 100.0):
            raise ValueError("true_percentile must be in [0, 100].")
        if not (0.0 <= fake_percentile <= 100.0):
            raise ValueError("fake_percentile must be in [0, 100].")

        heads_list, relations_list, tails_list, labels = data_w_label
        pos_scores = []
        neg_scores = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(
                self.discriminator.model.test_batch_size, heads_list, relations_list, tails_list, labels
            ):
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)

                batch_scores = self.discriminator.score(head_var, relation_var, tail_var).detach().cpu().numpy()
                batch_labels = np.asarray(batch_label)

                pos_mask = (batch_labels == 1)
                neg_mask = (batch_labels == 0)

                if np.any(pos_mask):
                    pos_scores.extend(batch_scores[pos_mask].reshape(-1).tolist())

                if np.any(neg_mask):
                    neg_scores.extend(batch_scores[neg_mask].reshape(-1).tolist())

        if n_generated_valid_negative > 0:
            labels_array = np.asarray(labels)
            pos_mask_all = (labels_array == 1)
            if np.any(pos_mask_all):
                pos_heads = np.asarray(heads_list)[pos_mask_all].astype(np.int64)
                pos_relations = np.asarray(relations_list)[pos_mask_all].astype(np.int64)
                pos_tails = np.asarray(tails_list)[pos_mask_all].astype(np.int64)

                pos_head_tensor = torch.LongTensor(pos_heads)
                pos_relation_tensor = torch.LongTensor(pos_relations)
                pos_tail_tensor = torch.LongTensor(pos_tails)

                valid_corrupter = BernCorrupterMulti(
                    (pos_heads.tolist(), pos_relations.tolist(), pos_tails.tolist()),
                    self.n_entity,
                    self.n_relation,
                    n_generated_valid_negative,
                )
                cand_head, cand_relation, cand_tail = valid_corrupter.corrupt(
                    pos_head_tensor,
                    pos_relation_tensor,
                    pos_tail_tensor,
                    keep_truth=False,
                )

                with torch.no_grad():
                    for _, batch_relation, _, batch_hs, batch_rs, batch_ts in batch_by_size(
                        self.discriminator.model.test_batch_size,
                        pos_head_tensor,
                        pos_relation_tensor,
                        pos_tail_tensor,
                        cand_head,
                        cand_relation,
                        cand_tail,
                        n_sample=len(pos_head_tensor),
                    ):
                        gen_step = self.generator.generator_step(
                            batch_hs,
                            batch_rs,
                            batch_ts,
                            n_sample=n_generated_valid_negative,
                            temperature=temperature,
                            train=False,
                        )
                        gen_head_fake, gen_tail_fake = next(gen_step)
                        gen_head_fake = gen_head_fake.to(config.device)
                        gen_tail_fake = gen_tail_fake.to(config.device)

                        batch_relation_device = batch_relation.to(config.device)
                        if gen_head_fake.dim() == batch_relation_device.dim() + 1:
                            relation_for_fake = batch_relation_device.unsqueeze(1).expand_as(gen_head_fake)
                        else:
                            relation_for_fake = batch_relation_device

                        gen_bad_scores = self.discriminator.score(
                            gen_head_fake,
                            relation_for_fake,
                            gen_tail_fake,
                        ).detach().cpu().numpy()

                        if gen_bad_scores.size > 0:
                            neg_scores.extend(gen_bad_scores.reshape(-1).tolist())
            else:
                logging.warning(
                    "Generator validation negatives not added: validation labels contain no positive samples."
                )

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            logging.warning("Validation threshold midpoint not updated: validation labels must contain both positive and negative samples.")
            return None

        true_stat = float(np.percentile(np.asarray(pos_scores), true_percentile))
        fake_stat = float(np.percentile(np.asarray(neg_scores), fake_percentile))
        logging.info(
            f"Validation percentile midpoint stats:\n"
            f"\ttrue_p{true_percentile:.1f}={true_stat:.6f}\n"
            f"\tfake_p{fake_percentile:.1f}={true_stat:.6f}"
        )
        return true_fake_balance * true_stat + (1 - true_fake_balance) * fake_stat

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                    filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.evaluate_on_ranking(test_data, heads, tails,
                                                        filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_triple_classification(self, test_data_w_label: tuple,
                                          optimizing_metric: str='accuracy',
                                          use_maxgood_minbad_threshold: bool=True) -> dict:
        if not isinstance(test_data_w_label[0], torch.Tensor):
            test_data_w_label = [torch.LongTensor(vec) for vec in test_data_w_label]
            
        print("Evaluating KBGAN discriminator on Triple Classification...")
        threshold = self.optimal_threshold if use_maxgood_minbad_threshold else None
        metrics = self.discriminator.evaluate_on_classification(test_data_w_label,
                                                                optimizing_metric=optimizing_metric,
                                                                is_threshold_tunning=False, external_threshold=threshold)
        return metrics