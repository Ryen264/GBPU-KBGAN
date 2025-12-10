import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from typing import Tuple
import logging
import os
import numpy as np

import config
from datasets import batch_by_size
from metrics import ranking_metrics, classification_metrics

class BaseModule(nn.Module):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__()
        
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.model_type = None      # to be set by subclasses, type: str
        self.model_config = None    # to be set by subclasses, type: config.Config

    def init_weight(self) -> None:
        pass

    def forward(self, head, relation, tail) -> torch.Tensor:
        pass

    def dist(self, head, relation, tail) -> torch.Tensor:
        pass

    def score(self, head, relation, tail) -> torch.Tensor:
        pass

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        pass

    def constraint(self) -> None:
        pass

    def prob(self, head, relation, tail) -> torch.Tensor:
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)

    def pair_loss(self, head, relation, tail, head_bad, tail_bad) -> torch.Tensor:
        d_good = self.dist(head, relation, tail)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, head, relation, tail, truth) -> torch.Tensor:
        probs = self.prob(head, relation, tail)
        n = probs.size(0)
        # Ensure indexing tensors are on same device as `probs`
        idx = torch.arange(0, n, device=probs.device, dtype=torch.long)
        truth = truth.to(probs.device)
        truth_probs = torch.log(probs[idx, truth] + 1e-30)
        return -truth_probs
    
class BaseModel(object):
    def __init__(self, n_entity: int, n_relation: int, use_gpu: bool = None):
        """
        BaseModel now supports selecting device and storing n_entity, n_relation, config at runtime.
        - If `use_gpu is None`, it will use the device selected by `config` module.
        - If `use_gpu is True`, it will attempt to use CUDA (and auto-select a GPU via config.select_gpu()).
        - If `use_gpu is False`, it will force CPU.
        """
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.model_type = None      # to be set by subclasses, type: str
        self.model_config = None    # to be set by subclasses, type: config.Config
        self.model_path = None      # to be set by subclasses, type: str
        self.model = None           # to be set when train/load, type: BaseModule
        self.weight_decay = 0
        self._set_device(use_gpu)

        self.task_dir = '.\\models\\' + config._config.dataset + '\\' + config._config.task + '\\components'
        os.makedirs(self.task_dir, exist_ok=True)

    def _set_device(self, use_gpu: bool = None) -> None:
        """Set runtime device for this BaseModel instance."""
        if use_gpu is None:
            # use device selected by config module
            try:
                self.device = config.device
            except Exception:
                self.device = torch.device('cpu')
        elif use_gpu:
            if torch.cuda.is_available():
                gpu_id = config.select_gpu()
                if gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    self.device = torch.device('cuda')
                logging.info('Using device %s', self.device)
            else:
                logging.warning('Requested GPU but CUDA is not available. Falling back to CPU.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            logging.info('Using device %s', self.device)

    def load(self, model_path) -> None:
        pass

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str]:
        pass

    def save(self) -> str:
        if self.model_path is None:
            raise ValueError("Model path is not set. Cannot save model.")
        
        try: 
            torch.save(self.model.state_dict(), self.model_path)
        except Exception as e:
            logging.error(f"Error saving model: {e}")
        return self.model_path

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def constraint(self) -> None:
        self.model.constraint()

    def _ensure_optimizer(self) -> None:
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)

    def is_trained_or_loaded(self) -> bool:
        return self.model is not None
    
    def get_score(self, head, relation, tail) -> torch.Tensor:
        return self.model.score(head, relation, tail)
    
    def get_prob_logit(self, head, relation, tail) -> torch.Tensor:
        return self.model.prob_logit(head, relation, tail)
    
    def get_pair_loss(self, head, relation, tail, head_bad, tail_bad) -> torch.Tensor:
        return self.model.pair_loss(head, relation, tail, head_bad, tail_bad)
    
    def evaluate_on_ranking(self, test_data, n_entity, heads, tails, filt=True, k_list=[1, 3, 10]) -> dict:
        mr_total = mrr_total = 0.0
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        test_data_no_label = test_data[:3]
        count = 0
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(config._config.test_batch_size, *test_data_no_label):
                batch_size = batch_head.size(0)

                all_var = torch.arange(0, n_entity).unsqueeze(0).expand(batch_size, n_entity).long().to(self.device)
                head_var = batch_head.unsqueeze(1).expand(batch_size, n_entity).to(self.device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, n_entity).to(self.device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, n_entity).to(self.device)

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
                            head_scores += heads[key_head].to(self.device) * 1e30
                            head_scores[head_id] = tmp
                            
                        key_tail = (head_id, relation_id)
                        if key_tail in tails and tails[key_tail]._nnz() > 1:
                            tmp = tail_scores[tail_id].item()
                            tail_scores += tails[key_tail].to(self.device) * 1e30
                            tail_scores[tail_id] = tmp

                    head_metrics = ranking_metrics(scores=head_scores, target=head_id, k_list=k_list)
                    tail_metrics = ranking_metrics(scores=tail_scores, target=tail_id, k_list=k_list)

                    head_mr = head_metrics['mr']
                    head_mrr = head_metrics['mrr']
                    head_hits = head_metrics['hits']

                    tail_mr = tail_metrics['mr']
                    tail_mrr = tail_metrics['mrr']
                    tail_hits = tail_metrics['hits']

                    mr_total += (head_mr + tail_mr)
                    mrr_total += (head_mrr + tail_mrr)
                    hits_total = [(hits_total[i] + head_hits[i] + tail_hits[i]) for i in range(len(k_list))]
                    count += 2
                    
        mr_rate = mr_total / count
        mrr_rate = mrr_total / count
        hits_rate = [hit_total / count for hit_total in hits_total]
        
        metrics = {}
        metrics['MR'] = mr_rate
        metrics['MRR'] = mrr_rate
        for i in range(len(k_list)):
            metrics[f'Hit@{k_list[i]}'] = hits_rate[i]

        metrics_str = f"Ranking metrics: {metrics}\n"
        logging.info(metrics_str)
        return metrics

    def evaluate_on_classification(self, test_data, optimizing_metric: str='accuracy') -> dict:        
        """
        Args:
            test_data: Tuple of (heads, relations, tails, labels)
            optimizing_metric: Metric to optimize when finding threshold ('accuracy', 'f1', etc.)
        """
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

            # Compute scores for all validation samples
            scores_list = []
            with torch.no_grad():
                for i in range(len(heads)):
                    head = torch.LongTensor([heads[i]]).to(config.device)
                    relation = torch.LongTensor([relations[i]]).to(config.device)
                    tail = torch.LongTensor([tails[i]]).to(config.device)

                    score = self.get_score(head, relation, tail)
                    scores_list.append(score)

            # Try different threshold values
            min_score = min(scores_list)
            max_score = max(scores_list)
            threshold_values = np.linspace(min_score, max_score, n_thresholds)

            best_val = 0.0
            best_threshold = 0.0

            # Determine if model is distance-based or similarity-based
            is_distance_based = self.model_type in ['TransE', 'TransD']

            for threshold in threshold_values:
                predictions = []
                for score in scores_list:
                    if is_distance_based:
                        predictions.append(1 if score < threshold else 0)
                    else:
                        predictions.append(1 if score > threshold else 0)
                
                metrics = classification_metrics(predictions, labels, scores=scores_list)
                val_metric = metrics.get(optimizing_metric, 0.0)
                
                if val_metric > best_val:
                    best_val = val_metric
                    best_threshold = threshold

            logging.info(f"Optimal threshold: {best_threshold:.4f} (metrics['{optimizing_metric}']={best_val:.4f})")
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
                head_var = batch_head.long().to(config.device)
                relation_var = batch_relation.long().to(config.device)
                tail_var = batch_tail.long().to(config.device)

                batch_scores = self.model.score(head_var, relation_var, tail_var)
                batch_scores = batch_scores.detach().cpu().tolist()

                scores_list.extend([float(s) for s in batch_scores])
                true_labels.extend([int(x) for x in batch_label])

        if len(scores_list) == 0:
            raise ValueError("No samples found in test_data for classification evaluation.")

        threshold = find_optimal_threshold(
            valid_data=(heads_list, relations_list, tails_list),
            labels=true_labels,
            n_thresholds=100
        )

        # determine whether smaller score means positive (distance-based models)
        is_distance_based = self.model_type in ['TransE', 'TransD']

        predictions = []
        for s in scores_list:
            if is_distance_based:
                predictions.append(1 if s < threshold else 0)
            else:
                predictions.append(1 if s > threshold else 0)

        metrics = classification_metrics(predictions, true_labels, scores=scores_list)
        metrics_str = f"Classification metrics: {metrics}\n"
        logging.info(metrics_str)
        return metrics