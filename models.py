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

class TransEModule(BaseModule):
    def __init__(self, n_entity, n_relation):
        super().__init__(n_entity, n_relation)
        self.model_type = 'TransE'
        self.model_config = config._config[self.model_type]

        self.p = self.model_config.p
        self.margin = self.model_config.margin
        self.temp = self.model_config.get('temp', 1)
        self.relation_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.entity_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head, relation, tail) -> torch.Tensor:
        return torch.norm(self.entity_embed(tail) - self.entity_embed(head) - self.relation_embed(relation) + 1e-30, p=self.p, dim=-1)

    def dist(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation ,tail) / self.temp

    def constraint(self):
        self.entity_embed.weight.data.renorm_(2, 0, 1)
        self.relation_embed.weight.data.renorm_(2, 0, 1)

class TransE(BaseModel):
    def __init__(self, n_entity, n_relation, use_gpu = None):
        super().__init__(n_entity, n_relation, use_gpu)
        self.model_type = 'TransE'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

    def load(self, model_path) -> None:
        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = TransEModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_path = model_path
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str | None]:
        if use_gpu is not None:
            self._set_device(use_gpu)

        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = TransEModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        head, relation, tail = train_data
        n_train = len(head)
        
        optimizer_class = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}.get(optimizer_name, Adam)
        optimizer = optimizer_class(self.model.parameters())

        best_perf = 0.0
        patience_counter = 0

        for epoch in range(self.n_epoch):
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx]
            relation = relation[rand_idx]
            tail = tail[rand_idx]

            head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
            head_cuda = head.to(self.device)
            relation_cuda = relation.to(self.device)
            tail_cuda = tail.to(self.device)
            head_corrupted = head_corrupted.to(self.device)
            tail_corrupted = tail_corrupted.to(self.device)
            epoch_loss = 0
            for h0, r, t0, h1, t1 in batch_by_num(self.n_batch, head_cuda, relation_cuda, tail_cuda,
                                                  head_corrupted, tail_corrupted, n_sample=n_train):
                self.zero_grad()
                loss = torch.sum(self.model.pair_loss(Variable(h0), Variable(r), Variable(t0), Variable(h1), Variable(t1)))
                loss.backward()
                optimizer.step()
                self.model.constraint()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                metrics = tester()
                test_perf = metrics['MRR']
                if (test_perf > best_perf):
                    if is_save_model:
                        print(f"Saving TransE at epoch {epoch + 1} with MRR {test_perf}.")
                        self.model_path = self.save()
                        print(f"Saved TransE successfully to: {self.model_path}")
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (use_early_stopping and patience_counter >= patience):
                    logging.info('Early stopping triggered at epoch %d (patience=%d)', epoch + 1, patience)
                    break
        if is_save_model:
            print(f"Saving trained TransE with best MRR {best_perf}.")
            self.model_path = self.save()
            print(f"Saved trained TransE successfully to: {self.model_path}")
            return best_perf, self.model_path
        return best_perf, None
    
class TransDModule(BaseModule):
    def __init__(self, n_entity, n_relation):
        super().__init__(n_entity, n_relation)
        self.model_type = 'TransD'
        self.model_config = config._config[self.model_type]

        self.margin = self.model_config.margin
        self.p = self.model_config.p
        self.temp = self.model_config.get('temp', 1)
        self.relation_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.entity_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.proj_relation_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.proj_entity_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, head, relation, tail) -> torch.Tensor:
        head_proj = self.entity_embed(head) +\
                   torch.sum(self.proj_entity_embed(head) * self.entity_embed(head), dim=-1, keepdim=True) * self.proj_relation_embed(relation)
        tail_proj = self.entity_embed(tail) +\
                   torch.sum(self.proj_entity_embed(tail) * self.entity_embed(tail), dim=-1, keepdim=True) * self.proj_relation_embed(relation)
        return torch.norm(tail_proj - self.relation_embed(relation) - head_proj + 1e-30, p=self.p, dim=-1)

    def dist(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def score(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation ,tail) / self.temp

    def constraint(self) -> None:
        for param in self.parameters():
            param.data.renorm_(2, 0, 1)

class TransD(BaseModel):
    def __init__(self, n_entity, n_relation, use_gpu = None):
        super().__init__(n_entity, n_relation, use_gpu)
        self.model_type = 'TransD'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.epoch_per_test = self.model_config.epoch_per_test

    def load_vec(self, vecpath) -> None:
        entity_mat = np.loadtxt(os.path.join(vecpath, 'entity2vec.vec'))
        self.model.entity_embed.weight.data.copy_(torch.from_numpy(entity_mat))

        relation_mat = np.loadtxt(os.path.join(vecpath, 'relation2vec.vec'))
        n_relation = relation_mat.shape[0]
        self.model.relation_embed.weight.data.copy_(torch.from_numpy(relation_mat))

        a_mat = np.loadtxt(os.path.join(vecpath, 'A.vec'))
        self.model.proj_relation_embed.weight.data.copy_(torch.from_numpy(a_mat[:n_relation, :]))
        self.model.proj_entity_embed.weight.data.copy_(torch.from_numpy(a_mat[n_relation:, :]))
        self.model.to(self.device)

    def load(self, model_path) -> None:
        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = TransDModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_path = model_path
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str | None]:
        if use_gpu is not None:
            self._set_device(use_gpu)
        
        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = TransDModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        head, relation, tail = train_data
        n_train = len(head)
        
        optimizer_class = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}.get(optimizer_name, Adam)
        optimizer = optimizer_class(self.model.parameters())

        best_perf = 0.0
        patience_counter = 0
        
        for epoch in range(self.n_epoch):
            rand_idx = torch.randperm(n_train)
            head = head[rand_idx]
            relation = relation[rand_idx]
            tail = tail[rand_idx]

            head_corrupted, tail_corrupted = corrupter.corrupt(head, relation, tail)
            head_cuda = head.to(self.device)
            relation_cuda = relation.to(self.device)
            tail_cuda = tail.to(self.device)
            head_corrupted = head_corrupted.to(self.device)
            tail_corrupted = tail_corrupted.to(self.device)
            epoch_loss = 0
            for h0, r, t0, h1, t1 in batch_by_num(self.n_batch, head_cuda, relation_cuda, tail_cuda,
                                                  head_corrupted, tail_corrupted, n_sample=n_train):
                self.zero_grad()
                loss = torch.sum(self.model.pair_loss(Variable(h0), Variable(r), Variable(t0), Variable(h1), Variable(t1)))
                loss.backward()

                optimizer.step()
                self.model.constraint()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                metrics = tester()
                test_perf = metrics['MRR']
                if (test_perf > best_perf):
                    if is_save_model:
                        print(f"Saving TransD at epoch {epoch + 1} with MRR {test_perf}.")
                        self.model_path = self.save()
                        print(f"Saved TransD successfully to: {self.model_path}")
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (use_early_stopping and patience_counter >= patience):
                    logging.info('Early stopping triggered at epoch %d', epoch + 1)
                    break
        if is_save_model:
            print(f"Saving trained TransD with best MRR {best_perf}.")
            self.model_path = self.save()
            print(f"Saved trained TransD successfully to: {self.model_path}")
            return best_perf, self.model_path
        return best_perf, None
    
class DistMultModule(BaseModule):
    def __init__(self, n_entity, n_relation):
        super().__init__(n_entity, n_relation)
        self.model_type = 'DistMult'
        self.model_config = config._config[self.model_type]

        sigma = 0.2
        self.relation_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.relation_embed.weight.data.div_((self.model_config.dim / sigma ** 2) ** (1 / 6))
        self.entity_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.entity_embed.weight.data.div_((self.model_config.dim / sigma ** 2) ** (1 / 6))

    def forward(self, head, relation, tail) -> torch.Tensor:
        return torch.sum(self.entity_embed(tail) * self.entity_embed(head) * self.relation_embed(relation), dim=-1)

    def dist(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation, tail)
    
    def score(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation, tail)

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

class DistMult(BaseModel):
    def __init__(self, n_entity, n_relation, use_gpu = None):
        super().__init__(n_entity, n_relation, use_gpu)
        self.model_type = 'DistMult'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.weight_decay = self.model_config.lam / self.model_config.n_batch
        self.sample_freq = self.model_config.sample_freq
        self.epoch_per_test = self.model_config.epoch_per_test

    def load(self, model_path) -> None:
        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = DistMultModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_path = model_path
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str | None]:
        if use_gpu is not None:
            self._set_device(use_gpu)

        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = DistMultModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        head, relation, tail = train_data
        n_train = len(head)
        
        optimizer_class = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}.get(optimizer_name, Adam)
        optimizer = optimizer_class(self.model.parameters(), weight_decay=self.weight_decay)
    
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
                head_corrupted = head_corrupted.to(self.device)
                relation_corrupted = relation_corrupted.to(self.device)
                tail_corrupted = tail_corrupted.to(self.device)

            for hs, rs, ts in batch_by_num(self.n_batch, head_corrupted, relation_corrupted, tail_corrupted, n_sample=n_train):
                self.zero_grad()
                label = torch.zeros(len(hs)).type(torch.LongTensor).to(self.device)
                loss = torch.sum(self.model.softmax_loss(Variable(hs), Variable(rs), Variable(ts), label))
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                metrics = tester()
                test_perf = metrics['MRR']
                if (test_perf > best_perf):
                    if is_save_model:
                        print(f"Saving DistMult at epoch {epoch + 1} with MRR {test_perf}.")
                        self.model_path = self.save()
                        print(f"Saved DistMult successfully to: {self.model_path}")
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (use_early_stopping and patience_counter >= patience):
                    logging.info('Early stopping triggered at epoch %d', epoch + 1)
                    break
        if is_save_model:
            print(f"Saving trained DistMult with best MRR {best_perf}.")
            self.model_path = self.save()
            print(f"Saved trained DistMult successfully to: {self.model_path}")
            return best_perf, self.model_path
        return best_perf, None

class ComplExModule(BaseModule):
    def __init__(self, n_entity, n_relation):
        super().__init__(n_entity, n_relation)
        self.model_type = 'ComplEx'
        self.model_config = config._config[self.model_type]

        self.sigma = 0.2
        self.dim = self.model_config.dim
        self.relation_re_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.relation_im_embed = nn.Embedding(n_relation, self.model_config.dim)
        self.entity_re_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.entity_im_embed = nn.Embedding(n_entity, self.model_config.dim)
        self.init_weight()

    def init_weight(self) -> None:
        for param in self.parameters():
            param.data.div_((self.dim / self.sigma ** 2) ** (1 / 6))

    def forward(self, head, relation, tail) -> torch.Tensor:
        return torch.sum(self.relation_re_embed(relation) * self.entity_re_embed(head) * self.entity_re_embed(tail), dim=-1) \
            + torch.sum(self.relation_re_embed(relation) * self.entity_im_embed(head) * self.entity_im_embed(tail), dim=-1) \
            + torch.sum(self.relation_im_embed(relation) * self.entity_re_embed(head) * self.entity_im_embed(tail), dim=-1) \
            - torch.sum(self.relation_im_embed(relation) * self.entity_im_embed(head) * self.entity_re_embed(tail), dim=-1)

    def dist(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation, tail)
    
    def score(self, head, relation, tail) -> torch.Tensor:
        return -self.forward(head, relation, tail)

    def prob_logit(self, head, relation, tail) -> torch.Tensor:
        return self.forward(head, relation, tail)

class ComplEx(BaseModel):
    def __init__(self, n_entity, n_relation, use_gpu = None):
        super().__init__(n_entity, n_relation, use_gpu)
        self.model_type = 'ComplEx'
        self.model_config = config._config[self.model_type]
        self.model_path = os.path.join(self.task_dir, self.model_config.model_file)

        self.n_epoch = self.model_config.n_epoch
        self.n_batch = self.model_config.n_batch
        self.weight_decay = self.model_config.lam / self.model_config.n_batch
        self.sample_freq = self.model_config.sample_freq
        self.epoch_per_test = self.model_config.epoch_per_test

    def load(self, model_path) -> None:
        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = ComplExModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_path = model_path
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str | None]:
        if use_gpu is not None:
            self._set_device(use_gpu)

        # Initialize model if not already trained or loaded
        if self.model is None:
            self.model = ComplExModule(self.n_entity, self.n_relation)
            self.model.to(self.device)

        head, relation, tail = train_data
        n_train = len(head)
        
        optimizer_class = {'Adam': Adam, 'SGD': SGD, 'AdamW': AdamW, 'RMSprop': RMSprop, 'Adagrad': Adagrad}.get(optimizer_name, Adam)
        optimizer = optimizer_class(self.model.parameters(), weight_decay=self.weight_decay)

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
                head_corrupted = head_corrupted.to(self.device)
                relation_corrupted = relation_corrupted.to(self.device)
                tail_corrupted = tail_corrupted.to(self.device)

            for hs, rs, ts in batch_by_num(self.n_batch, head_corrupted, relation_corrupted, tail_corrupted, n_sample=n_train):
                self.zero_grad()
                label = torch.zeros(len(hs)).type(torch.LongTensor).to(self.device)

                loss = torch.sum(self.model.softmax_loss(Variable(hs), Variable(rs), Variable(ts), label))
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()

            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, self.n_epoch, epoch_loss / n_train)
            if ((self.n_epoch >= self.epoch_per_test) and ((epoch + 1) % self.epoch_per_test == 0)):
                metrics = tester()
                test_perf = metrics['MRR']
                if (test_perf > best_perf):
                    if is_save_model:
                        print(f"Saving ComplEx at epoch {epoch + 1} with MRR {test_perf}.")
                        model_path = self.save()
                        print(f"Saved ComplEx successfully to: {model_path}")
                    best_perf = test_perf
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if (use_early_stopping and patience_counter >= patience):
                    logging.info('Early stopping triggered at epoch %d', epoch + 1)
                    break
        if is_save_model:
            print(f"Saving trained ComplEx with best MRR {best_perf}.")
            self.model_path = self.save()
            print(f"Saved trained ComplEx successfully to: {self.model_path}")
            return best_perf, self.model_path
        return best_perf, None