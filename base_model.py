from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from typing import Tuple, Optional
import logging
import os
import numpy as np

import config
from datasets import batch_by_size
from metrics import ranking_metrics, classification_metrics

class BaseModule(nn.Module):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__()
        
    def score(self, head, relation, tail) -> torch.Tensor:
        raise NotImplementedError
    
    def dist(self, head, relation, tail) -> torch.Tensor:
        raise NotImplementedError

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        raise NotImplementedError

    def prob(self, head, relation, tail) -> torch.Tensor:
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)
    
    def constraint(self) -> None:
        pass
    
    def pair_loss(self, head, relation, tail, head_bad, tail_bad) -> torch.Tensor:
        d_good = self.dist(head, relation, tail)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(self.margin + d_good - d_bad)
    
    def softmax_loss(self, head, relation, tail, truth) -> torch.Tensor:
        probs = self.prob(head, relation, tail)
        n = probs.size(0)
        row_idx = torch.arange(n, device=probs.device)
        truth_probs = torch.log(probs[row_idx, truth] + 1e-30)
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
        self.model_path = None 
        self.model = None # type: BaseModule
        self.weight_decay = 0
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task, 'components')
        print(f"Model will be saved to: {self.task_dir}")
        os.makedirs(self.task_dir, exist_ok=True)

    def save(self, filename) -> str:
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

    def gen_step(self, head, relation, tail, n_sample=1, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)

        # Forward pass: generate samples
        n, m = tail.size()
        relation_var = Variable(relation.to(self.device))
        head_var = Variable(head.to(self.device))
        tail_var = Variable(tail.to(self.device))

        logits = self.model.prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]

        # Yield samples to get rewards from discriminator
        rewards = yield sample_heads, sample_tails

        # Backward pass: update generator with REINFORCE
        if train:
            self.model.zero_grad()

            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()

            self.opt.step()
            self.model.constraint()
        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)
        head_var = Variable(src.to(self.device))
        relation_var = Variable(rel.to(self.device))
        tail_var = Variable(dst.to(self.device))

        head_fake_var = Variable(src_fake.to(self.device))
        tail_fake_var = Variable(dst_fake.to(self.device))
        losses = self.model.pair_loss(head_var, relation_var, tail_var, head_fake_var, tail_fake_var)
        fake_scores = self.model.score(head_fake_var, relation_var, tail_fake_var)

        if train:
            self.model.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.model .constraint()
        return losses.data, -fake_scores.data

    def evaluate_on_ranking(self, test_data, n_entity, heads, tails, filt=True, k_list=None) -> dict:
        if k_list is None:
            k_list = [1, 3, 10]
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

            # Compute scores for all validation samples (batched for efficiency)
            scores_list = []
            with torch.no_grad():
                for batch_head, batch_relation, batch_tail in batch_by_size(config._config.test_batch_size,
                                                                           heads, relations, tails):
                    head_var = torch.LongTensor(batch_head).to(self.device)
                    relation_var = torch.LongTensor(batch_relation).to(self.device)
                    tail_var = torch.LongTensor(batch_tail).to(self.device)

                    batch_scores = self.model.score(head_var, relation_var, tail_var)
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
            is_distance_based = self.model.is_distance_based

            for threshold in threshold_values:
                if is_distance_based:
                    predictions = np.where(scores_array < threshold, 1, 0).tolist()
                else:
                    predictions = np.where(scores_array > threshold, 1, 0).tolist()
                
                metrics = classification_metrics(predictions, labels, scores=scores_list)
                val_metric = metrics.get(optimizing_metric, 0.0)
                
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
                head_var = torch.LongTensor(batch_head).to(self.device)
                relation_var = torch.LongTensor(batch_relation).to(self.device)
                tail_var = torch.LongTensor(batch_tail).to(self.device)

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
        is_distance_based = self.model.is_distance_based

        # Vectorized prediction generation
        scores_array = np.array(scores_list)
        if is_distance_based:
            predictions = np.where(scores_array < threshold, 1, 0).tolist()
        else:
            predictions = np.where(scores_array > threshold, 1, 0).tolist()

        # For distance-based models lower scores mean better; invert for AUC so higher is better
        scores_for_auc = [-s for s in scores_list] if is_distance_based else scores_list

        metrics = classification_metrics(predictions, true_labels, scores=scores_for_auc)
        # Format metrics for cleaner output
        metrics_display = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
        metrics_str = f"Classification metrics: {metrics_display}\n"
        logging.info(metrics_str)
        return metrics
    
    