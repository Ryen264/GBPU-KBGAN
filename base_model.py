from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
import logging
import os

import config
from datasets import batch_by_size
from metrics import ranking_metrics, mrr_mr_hitk

class BaseModule(nn.Module):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__()
        
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return nnf.softmax(self.prob_logit(head, relation, tail), dim=-1)
    
    def constraint(self) -> None:
        pass
    
    def pair_loss(self, head_good: torch.Tensor, relation: torch.Tensor, tail_good: torch.Tensor,
                  head_bad: torch.Tensor, tail_bad: torch.Tensor) -> torch.Tensor:
        d_good = self.dist(head_good, relation, tail_good)
        d_bad = self.dist(head_bad, relation, tail_bad)
        return nnf.relu(d_good - d_bad + self.margin)
    
    def softmax_loss(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
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
        self.model = None           # type: BaseModule
        self.weight_decay = 0
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task, 'components')
        os.makedirs(self.task_dir, exist_ok=True)

    def save(self, filename: str) -> str:
        torch.save(self.model.state_dict(), filename)
        return filename

    def load(self, filename: str) -> None:
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

    def gen_step(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
                 n_sample: int=1, temperature: float=1.0, train: bool=True):
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

    def dis_step(self, head_good: torch.Tensor, relation: torch.Tensor, tail_good: torch.Tensor,
                 head_bad: torch.Tensor, tail_bad: torch.Tensor, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)
        head_var = Variable(head_good.to(self.device))
        relation_var = Variable(relation.to(self.device))
        tail_var = Variable(tail_good.to(self.device))

        head_bad_var = Variable(head_bad.to(self.device))
        tail_bad_var = Variable(tail_bad.to(self.device))
        losses = self.model.pair_loss(head_var, relation_var, tail_var, head_bad_var, tail_bad_var)
        fake_scores = self.model.score(head_bad_var, relation_var, tail_bad_var)

        if train:
            self.model.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.model.constraint()
        return losses.data, -fake_scores.data
    
    def test_link(self, test_data: list, n_entity: int, heads: dict, tails: dict, filt: bool=True) -> float:
        mrr_total = 0
        mr_total = 0
        hit10_total = 0
        count = 0
        with torch.no_grad():  # Thay volatile=True
            for batch_s, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):
                batch_size = batch_s.size(0)
                
                # Không cần Variable, tensor bình thường là được
                rel_var = batch_r.unsqueeze(1).expand(batch_size, n_entity).cuda()
                src_var = batch_s.unsqueeze(1).expand(batch_size, n_entity).cuda()
                dst_var = batch_t.unsqueeze(1).expand(batch_size, n_entity).cuda()
                
                all_var = torch.arange(0, n_entity).unsqueeze(0).expand(batch_size, n_entity).long().cuda()
                
                # Tính điểm
                batch_dst_scores = self.mdl.score(src_var, rel_var, all_var)
                batch_src_scores = self.mdl.score(all_var, rel_var, dst_var)
                
                # Convert to numpy if needed
                batch_dst_scores = batch_dst_scores.detach()
                batch_src_scores = batch_src_scores.detach()

                for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                    s_id, r_id, t_id = s.item(), r.item(), t.item()
                    
                    if filt:
                        key_dst = (s_id, r_id)
                        key_src = (t_id, r_id)
                        
                        if key_dst in tails and tails[key_dst]._nnz() > 1:
                            tmp = dst_scores[t_id].item()
                            dst_scores += tails[key_dst].cuda() * 1e30
                            dst_scores[t_id] = tmp
                        
                        if key_src in heads and heads[key_src]._nnz() > 1:
                            tmp = src_scores[s_id].item()
                            src_scores += heads[key_src].cuda() * 1e30
                            src_scores[s_id] = tmp
                    
                    mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t_id)
                    mrr_total += mrr
                    mr_total += mr
                    hit10_total += hit10

                    mrr, mr, hit10 = mrr_mr_hitk(src_scores, s_id)
                    mrr_total += mrr
                    mr_total += mr
                    hit10_total += hit10

                    count += 2

        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', 
                    mrr_total / count, mr_total / count, hit10_total / count)
        return mrr_total / count
    
    def evaluate_on_ranking(self, test_data: list, n_entity: int, heads: dict, tails: dict, filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
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