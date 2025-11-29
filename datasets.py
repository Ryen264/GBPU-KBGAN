from random import randint
from collections import defaultdict
import torch
import numpy as np
from numpy.random import choice

def sparse_heads_tails(n_entity, train_data, valid_data=None, test_data=None):
    if train_data:
        train_head, train_relation, train_tail = train_data
    else:
        train_head = train_relation = train_tail = []
    if valid_data:
        valid_head, valid_relation, valid_tail = valid_data
    else:
        valid_head = valid_relation = valid_tail = []
        
    if test_data:
        test_head, test_relation, test_tail = test_data
    else:
        test_head = test_relation = test_tail = []
        
    all_head = train_head + valid_head + test_head
    all_relation = train_relation + valid_relation + test_relation
    all_tail = train_tail + valid_tail + test_tail
    
    heads = defaultdict(lambda: set())
    tails = defaultdict(lambda: set())
    for h, r, t in zip(all_head, all_relation, all_tail):
        heads[(t, r)].add(h)
        tails[(h, r)].add(t)
    
    heads_sparse = tails_sparse = {}
    for k in heads.keys():
        indices = torch.LongTensor([list(heads[k])])
        values = torch.ones(len(heads[k]))
        heads_sparse[k] = torch.sparse_coo_tensor(indices, values, torch.Size([n_entity]), dtype=torch.float32)
    for k in tails.keys():
        indices = torch.LongTensor([list(tails[k])])
        values = torch.ones(len(tails[k]))
        tails_sparse[k] = torch.sparse_coo_tensor(indices, values, torch.Size([n_entity]), dtype=torch.float32)
    return heads_sparse, tails_sparse

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(randint(0, i+1))
    for ls in lists:
        for i, item in enumerate(ls):
            j = idx[i]
            ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    for i in range(n_batch):
        head = int(n_sample * i / n_batch)
        tail = int(n_sample * (i + 1) / n_batch)
        ret = [ls[head:tail] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])
    head = 0
    while head < n_sample:
        tail = min(n_sample, head + batch_size)
        ret = [ls[head:tail] for ls in lists]
        head += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def get_bern_prob(data, n_relation):
    head, relation, tail = data
    edges = defaultdict(lambda: defaultdict(lambda: set()))
    rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
    for s, r, t in zip(head, relation, tail):
        edges[r][s].add(t)
        rev_edges[r][t].add(s)
    bern_prob = torch.zeros(n_relation)
    for r in edges.keys():
        tph = sum(len(tails) for tails in edges[r].values()) / len(edges[r])
        htp = sum(len(heads) for heads in rev_edges[r].values()) / len(rev_edges[r])
        bern_prob[r] = tph / (tph + htp)
    return bern_prob

class BernCorrupter(object):
    def __init__(self, data, n_entity, n_relation):
        self.bern_prob = get_bern_prob(data, n_relation)
        self.n_entity = n_entity

    def corrupt(self, head, relation, tail):
        prob = self.bern_prob[relation]
        selection = torch.bernoulli(prob).numpy().astype('int64')
        entity_random = choice(self.n_entity, len(head))
        head_out = (1 - selection) * head.numpy() + selection * entity_random
        tail_out = selection * tail.numpy() + (1 - selection) * entity_random
        return torch.from_numpy(head_out), torch.from_numpy(tail_out)

class BernCorrupterMulti(object):
    def __init__(self, data, n_entity, n_relation, n_sample):
        self.bern_prob = get_bern_prob(data, n_relation)
        self.n_entity = n_entity
        self.n_sample = n_sample

    def corrupt(self, head, relation, tail, keep_truth=True):
        n = len(head)
        prob = self.bern_prob[relation]
        selection = torch.bernoulli(prob).numpy().astype('bool')
        head_out = np.tile(head.numpy(), (self.n_sample, 1)).transpose()
        tail_out = np.tile(tail.numpy(), (self.n_sample, 1)).transpose()
        relation_out = relation.unsqueeze(1).expand(n, self.n_sample)
        if keep_truth:
            entity_random = choice(self.n_entity, (n, self.n_sample - 1))
            head_out[selection, 1:] = entity_random[selection]
            tail_out[~selection, 1:] = entity_random[~selection]
        else:
            entity_random = choice(self.n_entity, (n, self.n_sample))
            head_out[selection, :] = entity_random[selection]
            tail_out[~selection, :] = entity_random[~selection]
        return torch.from_numpy(head_out), relation_out, torch.from_numpy(tail_out)