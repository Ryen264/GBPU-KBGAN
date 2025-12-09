import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from typing import Tuple
import logging

from config import config
import config as _config_module
from datasets import batch_by_size
from metrics import ranking_metrics

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

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
    def __init__(self, use_gpu: bool = None):
        """
        BaseModel now supports selecting device at runtime.
        - If `use_gpu is None`, it will use the device selected by `config` module.
        - If `use_gpu is True`, it will attempt to use CUDA (and auto-select a GPU via config.select_gpu()).
        - If `use_gpu is False`, it will force CPU.
        """
        self.model = None # type: BaseModule
        self.weight_decay = 0
        self._set_device(use_gpu)

    def _set_device(self, use_gpu: bool = None) -> None:
        """Set runtime device for this BaseModel instance."""
        if use_gpu is None:
            # use device selected by config module
            try:
                self.device = _config_module.device
            except Exception:
                self.device = torch.device('cpu')
        elif use_gpu:
            if torch.cuda.is_available():
                gpu_id = _config_module.select_gpu()
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

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam', use_gpu: bool = None) -> Tuple[float, str]:
        # allow overriding device for training
        if use_gpu is not None:
            self._set_device(use_gpu)
        pass

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def constraint(self) -> None:
        self.model.constraint()
        
    def save(self, model_path) -> None:
        torch.save(self.model.state_dict(), model_path)
        
    def load(self, model_path) -> None:
        # Ensure model is loaded onto the chosen device
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def _ensure_optimizer(self) -> None:
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)
    
    def get_score(self, head, relation, tail) -> torch.Tensor:
        return self.model.score(head, relation, tail)
    
    def get_prob_logit(self, head, relation, tail) -> torch.Tensor:
        return self.model.prob_logit(head, relation, tail)
    
    def get_pair_loss(self, head, relation, tail, head_bad, tail_bad) -> torch.Tensor:
        return self.model.pair_loss(head, relation, tail, head_bad, tail_bad)
    
    def evaluate(self, test_data, n_entity, heads, tails, filt=True, k_list=[1, 3, 10]) -> dict:
        """
        Evaluate the model on Link Prediction task.
        """
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        test_data_no_label = test_data[:3]
        count = 0
        with torch.no_grad():  # Thay volatile=True
            for batch_head, batch_relation, batch_tail in batch_by_size(config().test_batch_size, *test_data_no_label):
                batch_size = batch_head.size(0)

                all_var = torch.arange(0, n_entity).unsqueeze(0).expand(batch_size, n_entity).long().to(self.device)
                head_var = batch_head.unsqueeze(1).expand(batch_size, n_entity).to(self.device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, n_entity).to(self.device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, n_entity).to(self.device)

                batch_head_scores = self.model.score(all_var, relation_var, tail_var)
                batch_tail_scores = self.model.score(head_var, relation_var, all_var)
            
                # Convert to numpy if needed
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
                    head_target_score = head_metrics.get('target_score')

                    tail_mr = tail_metrics['mr']
                    tail_mrr = tail_metrics['mrr']
                    tail_hits = tail_metrics['hits']
                    tail_target_score = tail_metrics.get('target_score')

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
        metrics_str = f"MR = {mr_rate}\nMRR = {mrr_rate}\n"
        for i in range(len(k_list)):
            metrics[f'Hit@{k_list[i]}'] = hits_rate[i]
            metrics_str += f"Hit@{k_list[i]} = {hits_rate[i]}\n"

        logging.info(metrics_str)
        return metrics