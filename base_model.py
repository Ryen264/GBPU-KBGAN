import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from typing import Tuple
import logging

from config import config, device
from datasets import batch_by_size

def mr_mrr_hitsk(scores, target, k_list=[1, 3, 10]):
    _, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target
    target_rank = torch.nonzero(find_target)[0, 0] + 1
    target_score = scores[target].item()  # Get the score of the target entity
    return target_rank, 1 / target_rank, [int(target_rank <= k) for k in k_list], target_score

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weight(self):
        pass

    def forward(self, head, relation, tail):
        pass

    def dist(self, head, relation, tail):
        pass

    def score(self, head, relation, tail):
        pass

    def prob_logit(self, head, relation, tail):
        pass

    def constraint(self):
        pass

    def prob(self, head, relation, tail):
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)

    def pair_loss(self, head, relation, tail, head_bad, tail_bad):
        d_good = self.dist(head, relation, tail)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, head, relation, tail, truth):
        probs = self.prob(head, relation, tail)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).to(device), truth] + 1e-30)
        return -truth_probs
    
class BaseModel(object):
    def __init__(self):
        self.model = None # type: BaseModule
        self.weight_decay = 0

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam') -> Tuple[float, str]:
        pass
    def zero_grad(self):
        self.model.zero_grad()
    def constraint(self):
        self.model.constraint()
    def save_model(self, model_filename):
        torch.save(self.model.state_dict(), model_filename)

    def load_model(self, model_filename):
        state_dict = torch.load(model_filename, map_location=lambda storage, location: storage.to(device), weights_only=True)
        self.model.load_state_dict(state_dict)

    def _ensure_optimizer(self):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)
    
    def get_score(self, head, relation, tail):
        return self.model.score(head, relation, tail)
    def get_prob_logit(self, head, relation, tail):
        return self.model.prob_logit(head, relation, tail)
    def get_pair_loss(self, head, relation, tail, head_bad, tail_bad):
        return self.model.pair_loss(head, relation, tail, head_bad, tail_bad)
    def evaluate(self, test_data, n_entity, heads, tails, filt=True) -> dict:
        """
        Evaluate the model on Link Prediction task.
        """
        mr_total = mrr_total = 0.0
        k_list = [1, 3, 10]
        hits_total = [0] * len(k_list)

        count = 0
        with torch.no_grad():  # Thay volatile=True
            for batch_head, batch_relation, batch_tail in batch_by_size(config().test_batch_size, *test_data):
                batch_size = batch_head.size(0)

                all_var = torch.arange(0, n_entity).unsqueeze(0).expand(batch_size, n_entity).long().to(device)
                head_var = batch_head.unsqueeze(1).expand(batch_size, n_entity).to(device)
                relation_var = batch_relation.unsqueeze(1).expand(batch_size, n_entity).to(device)
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, n_entity).to(device)

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
                            head_scores += heads[key_head].to(device) * 1e30
                            head_scores[head_id] = tmp
                            
                        key_tail = (head_id, relation_id)
                        if key_tail in tails and tails[key_tail]._nnz() > 1:
                            tmp = tail_scores[tail_id].item()
                            tail_scores += tails[key_tail].to(device) * 1e30
                            tail_scores[tail_id] = tmp

                    head_mr, head_mrr, head_hits, head_target_score = mr_mrr_hitsk(scores=head_scores, target=head_id, k_list=k_list)
                    tail_mr, tail_mrr, tail_hits, tail_target_score = mr_mrr_hitsk(scores=tail_scores, target=tail_id, k_list=k_list)                    
                    
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