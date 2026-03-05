import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
from torch.autograd import Variable
from typing import Tuple
import logging
import os
import numpy as np
import config

from datasets import batch_by_num
from base_model import BaseModel, BaseModule

EPSILON = 1e-30
OPTIMIZER_MAP = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}

class TransEModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'TransE'

        self.dim = config.dim
        self.margin = config.margin
        self.p = config.p
        self.temp = config.temp

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.entity_embed(tail) - self.entity_embed(head) - self.relation_embed(relation) + EPSILON, p=self.p, dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation ,tail) / self.temp

    def constraint(self) -> None:
        self.entity_embed.weight.data.renorm_(2, 0, 1)
        self.relation_embed.weight.data.renorm_(2, 0, 1)

class TransE(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'TransE'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate

        self.model = TransEModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.margin = self.model.margin
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr)

    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> float:
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        patience_counter = 0
        for epoch in range(self.n_epoch):
            epoch_loss = 0
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx]
            relation = relation[rand_idx]
            tail = tail[rand_idx]
            head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
            head_device = head.to(config.device)
            relation_device = relation.to(config.device)
            tail_device = tail.to(config.device)
            head_corrupted = head_corrupted.to(config.device)
            tail_corrupted = tail_corrupted.to(config.device)
            
            for h0, r, t0, h1, t1 in batch_by_num(self.n_batch, head_device, relation_device, tail_device,
                                                  head_corrupted, tail_corrupted, n_sample=n_train):
                self.model.zero_grad()
                loss = torch.sum(self.model.pair_loss(h0, r, t0, h1, t1))
                loss.backward()
                self.opt.step()
                self.model.constraint()
                epoch_loss += loss.item()
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()
                if (test_perf > best_perf):
                    self.save()
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, early_stop_patience)
                    break
        return best_perf
    
class TransDModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'TransD'

        self.dim = config.dim
        self.margin = config.margin
        self.p = config.p
        self.temp = config.temp
        
        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.proj_relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.proj_entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.is_distance_based = True
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_proj = self.entity_embed(head) +\
                   torch.sum(self.proj_entity_embed(head) * self.entity_embed(head), dim=-1, keepdim=True) * self.proj_relation_embed(relation)
        tail_proj = self.entity_embed(tail) +\
                   torch.sum(self.proj_entity_embed(tail) * self.entity_embed(tail), dim=-1, keepdim=True) * self.proj_relation_embed(relation)
        return torch.norm(tail_proj - self.relation_embed(relation) - head_proj + EPSILON, p=self.p, dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail) / self.temp

    def constraint(self) -> None:
        for param in self.parameters():
            param.data.renorm_(2, 0, 1)

class TransD(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'TransD'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate

        self.model = TransDModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.margin = self.model.margin
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr)

    def load_vec(self, vecpath: str) -> None:
        entity_mat = np.loadtxt(os.path.join(vecpath, 'entity2vec.vec'))
        self.model.entity_embed.weight.data.copy_(torch.from_numpy(entity_mat))
        relation_mat = np.loadtxt(os.path.join(vecpath, 'relation2vec.vec'))
        n_relation = relation_mat.shape[0]
        self.model.relation_embed.weight.data.copy_(torch.from_numpy(relation_mat))
        a_mat = np.loadtxt(os.path.join(vecpath, 'A.vec'))
        self.model.proj_relation_embed.weight.data.copy_(torch.from_numpy(a_mat[:n_relation, :]))
        self.model.proj_entity_embed.weight.data.copy_(torch.from_numpy(a_mat[n_relation:, :]))
        self.model.to(config.device)

    def train(self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> float:
        head, relation, tail = train_data
        n_train = len(head)        
        best_perf = 0.0
        patience_counter = 0
        for epoch in range(self.n_epoch):
            epoch_loss = 0
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx]
            relation = relation[rand_idx]
            tail = tail[rand_idx]
            head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
            head_device = head.to(config.device)
            relation_device = relation.to(config.device)
            tail_device = tail.to(config.device)
            head_corrupted = head_corrupted.to(config.device)
            tail_corrupted = tail_corrupted.to(config.device)
            
            for h0, r, t0, h1, t1 in batch_by_num(self.n_batch, head_device, relation_device, tail_device,
                                                  head_corrupted, tail_corrupted, n_sample=n_train):
                self.model.zero_grad()
                loss = torch.sum(self.model.pair_loss(Variable(h0), Variable(r), Variable(t0), Variable(h1), Variable(t1)))
                loss.backward()
                self.opt.step()
                self.model.constraint()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()
                if (test_perf > best_perf):
                    self.save()
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                    logging.info('Early stopping triggered at epoch %d', epoch + 1)
                    break
        return best_perf
    
class DistMultModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'DistMult'
        
        self.dim = config.dim
        self.sigma = config.sigma

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_embed = nn.Embedding(self.n_relation, self.dim)
        self.relation_embed.weight.data.div_((self.dim / self.sigma ** 2) ** (1 / 6))
        self.entity_embed = nn.Embedding(self.n_entity, self.dim)
        self.entity_embed.weight.data.div_((self.dim / self.sigma ** 2) ** (1 / 6))
        self.is_distance_based = False

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.entity_embed(tail) * self.entity_embed(head) * self.relation_embed(relation), dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail)
    
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

class DistMult(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DistMult'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)
        
        self.lam = self.model_config.lam
        self.n_sample = self.model_config.n_sample
        self.sample_freq = self.model_config.sample_freq
        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate
        self.weight_decay = self.lam / self.n_batch

        self.model = DistMultModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> float:
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        patience_counter = 0
        for epoch in range(self.n_epoch):
            epoch_loss = 0
            if (epoch % self.sample_freq == 0):
                rand_idx = torch.randperm(n_train)
                head = head[rand_idx]
                relation = relation[rand_idx]
                tail = tail[rand_idx]
                head_corrupted, relation_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
                head_corrupted = head_corrupted.to(config.device)
                relation_corrupted = relation_corrupted.to(config.device)
                tail_corrupted = tail_corrupted.to(config.device)

            for hs, rs, ts in batch_by_num(self.n_batch, head_corrupted, relation_corrupted, tail_corrupted, n_sample=n_train):
                self.model.zero_grad()
                label = torch.zeros(len(hs)).type(torch.LongTensor).to(config.device)
                hs_var, rs_var, ts_var = Variable(hs), Variable(rs), Variable(ts)
                softmax_loss = self.model.softmax_loss(hs_var, rs_var, ts_var, label)
                loss = torch.sum(softmax_loss)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()
                if (test_perf > best_perf):
                    self.save()
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, early_stop_patience)
                    break
        return best_perf

class ComplExModule(BaseModule):
    def __init__(self, n_entity: int, n_relation: int, config: config.config):
        super().__init__()
        self.model_type = 'ComplEx'

        self.dim = config.dim
        self.sigma = config.sigma

        self.n_entity, self.n_relation = n_entity, n_relation
        self.relation_re_embed = nn.Embedding(self.n_relation, self.dim)
        self.relation_im_embed = nn.Embedding(self.n_relation, self.dim)
        self.entity_re_embed = nn.Embedding(self.n_entity, self.dim)
        self.entity_im_embed = nn.Embedding(self.n_entity, self.dim)
        self.is_distance_based = False
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.div_((self.dim / self.sigma ** 2) ** (1 / 6))

    def forward(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.relation_re_embed(relation) * self.entity_re_embed(head) * self.entity_re_embed(tail), dim=-1) \
            + torch.sum(self.relation_re_embed(relation) * self.entity_im_embed(head) * self.entity_im_embed(tail), dim=-1) \
            + torch.sum(self.relation_im_embed(relation) * self.entity_re_embed(head) * self.entity_im_embed(tail), dim=-1) \
            - torch.sum(self.relation_im_embed(relation) * self.entity_im_embed(head) * self.entity_re_embed(tail), dim=-1)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail)
    
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return -self.forward(head, relation, tail)

    def prob_logit(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return self.forward(head, relation, tail)

class ComplEx(BaseModel):
    def __init__(self, n_entity: int, n_relation: int):
        super().__init__(n_entity, n_relation)
        self.model_type = 'ComplEx'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)
        
        self.lam = self.model_config.lam
        self.n_sample = self.model_config.n_sample
        self.sample_freq = self.model_config.sample_freq
        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

        self.optimizer_name = self.model_config.optimizer
        self.lr = self.model_config.learning_rate
        self.weight_decay = self.lam / self.n_batch

        self.model = ComplExModule(self.n_entity, self.n_relation, self.model_config)
        self.model.to(config.device)
        self.is_distance_based = self.model.is_distance_based
        self.opt = OPTIMIZER_MAP[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              corrupter, tester, early_stop_patience: int=-1) -> float:            
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        patience_counter = 0
        for epoch in range(self.n_epoch):
            epoch_loss = 0
            if (epoch % self.sample_freq == 0):
                rand_idx = torch.randperm(n_train)
                head = head[rand_idx]
                relation = relation[rand_idx]
                tail = tail[rand_idx]
                head_corrupted, relation_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
                head_corrupted = head_corrupted.to(config.device)
                relation_corrupted = relation_corrupted.to(config.device)
                tail_corrupted = tail_corrupted.to(config.device)

            for hs, rs, ts in batch_by_num(self.n_batch, head_corrupted, relation_corrupted, tail_corrupted, n_sample=n_train):
                self.model.zero_grad()
                label = torch.zeros(len(hs)).type(torch.LongTensor).to(config.device)
                hs_var, rs_var, ts_var = Variable(hs), Variable(rs), Variable(ts)
                softmax_loss = self.model.softmax_loss(hs_var, rs_var, ts_var, label)
                loss = torch.sum(softmax_loss)
                loss.backward()
                self.opt.step()
                epoch_loss += loss.item()
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                test_perf = tester()               
                if (test_perf > best_perf):
                    self.save()
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (early_stop_patience > 0 and patience_counter >= early_stop_patience):
                    logging.info('Early stopping triggered at epoch %d', epoch + 1)
                    break
        return best_perf