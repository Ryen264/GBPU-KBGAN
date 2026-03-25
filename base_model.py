from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
import logging
import os

import config
from datasets import batch_by_size
from metrics import ranking_metrics

EPSILON = 1e-30
FILTER_RANKING_PENALTY = 1e30

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.is_distance_based = None
        self.margin = None
        
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)
    
    def ensure_optimizer(self) -> None:
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.parameters(), weight_decay=0.0)

    def constraint(self) -> None:
        pass

    def parameters(self, recurse = True):
        return super().parameters(recurse)

    def pair_loss(self, head_good: torch.Tensor, relation: torch.Tensor, tail_good: torch.Tensor,
                  head_bad: torch.Tensor, tail_bad: torch.Tensor) -> torch.Tensor:
        if not self.is_distance_based:
            raise NotImplementedError("Pairwise loss is only implemented for distance-based models within margin.")
        
        d_good = self.dist(head_good, relation, tail_good)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(d_good - d_bad + self.margin)
    
    def softmax_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        probs = self.prob(head, relation, tail)
        n = probs.size(0)
        device = probs.device

        row_idx = torch.arange(n, device=device)
        truth_probs = torch.log(probs[row_idx, truth] + EPSILON)
        return -truth_probs
    
class BaseModel(object):
    def __init__(self, n_entity: int, n_relation: int):
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.model_path = None 
        self.weight_decay = 0.0
        self.model = None           # type: BaseModule
        self.opt = None             # type: torch.optim.Optimizer
        self.lr = 0.0

        self.dataset = config._config.dataset
        self.task = config._config.task
        self.task_dir = os.path.join('.', 'models', self.dataset, self.task, 'components')
        os.makedirs(self.task_dir, exist_ok=True)
        self.test_batch_size = config._config.test_batch_size

    def load(self, filepath: str) -> None:
        self.model.load_state_dict(torch.load(filepath, map_location=config.device))

    def save(self, filepath: str=None) -> None:
        if filepath is None:
            filepath = self.model_path
        torch.save(self.model.state_dict(), filepath)

    def ensure_optimizer(self) -> None:
        self.model.ensure_optimizer()

    def constraint(self) -> None:
        self.model.constraint()

    def parameters(self, recurse = True):
        return self.model.parameters(recurse)
    
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.score(head, relation, tail)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.dist(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.model.prob_logit(head, relation, tail)

    def gen_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                 n_sample: int=1, temperature: float=1.0, train: bool=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)

        # Forward pass: generate samples
        n, m = tail.size()
        relation_var = Variable(relation.to(config.device))
        head_var = Variable(head.to(config.device))
        tail_var = Variable(tail.to(config.device))

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

    def dis_step(self, head_good: torch.Tensor, relation: torch.Tensor, tail_good: torch.Tensor,
                 head_bad: torch.Tensor, tail_bad: torch.Tensor, train: bool=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)

        head_good_var = Variable(head_good.to(config.device))
        relation_var = Variable(relation.to(config.device))
        tail_good_var = Variable(tail_good.to(config.device))
        head_bad_var = Variable(head_bad.to(config.device))
        tail_bad_var = Variable(tail_bad.to(config.device))

        losses = self.model.pair_loss(head_good_var, relation_var, tail_good_var, head_bad_var, tail_bad_var)
        fake_scores = self.model.score(head_bad_var, relation_var, tail_bad_var)

        if train:
            self.model.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.model.constraint()
        return losses.data, -fake_scores.data
    
    def test_link(self, test_data: list, heads: dict, tails: dict, filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        test_data_no_label = test_data[:3]
        count = 0
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(self.test_batch_size, *test_data_no_label):
                batch_size = batch_head.size(0)

                head_var = batch_head.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                all_var = torch.arange(0, self.n_entity).unsqueeze(0).expand(batch_size, self.n_entity).long().to(config.device)

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

                    head_metrics = ranking_metrics(head_scores, head_id, k_list=k_list)
                    tail_metrics = ranking_metrics(tail_scores, tail_id, k_list=k_list)

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
        metrics['mr'] = mr_rate
        metrics['mrr'] = mrr_rate
        for i in range(len(k_list)):
            metrics[f'hit@{k_list[i]}'] = hits_rate[i]

        # Format metrics for cleaner output
        parts = []
        label_map = {'mr': 'MR', 'mrr': 'MRR'}
        for k, v in metrics.items():
            label = label_map.get(k, k.replace('hit@', 'Hit@'))
            parts.append(f"{label}: {v:.4f}")
        metrics_str = f"Ranking metrics: {', '.join(parts)}\n"
        logging.info(metrics_str)
        return metrics