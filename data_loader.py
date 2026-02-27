from itertools import count
from collections import namedtuple
from typing import Union
import logging

KBIndex = namedtuple('KBIndex', ['entity_list', 'relation_list', 'entity_id', 'relation_id'])

def index_entity_relation(*filenames: str) -> KBIndex:
    entity_set = set()
    relation_set = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                s, r, t = line.strip().split('\t')[:3]
                entity_set.add(s)
                entity_set.add(t)
                relation_set.add(r)
    entity_list = sorted(list(entity_set))
    relation_list = sorted(list(relation_set))
    entity_id = dict(zip(entity_list, count()))
    relation_id = dict(zip(relation_list, count()))
    return KBIndex(entity_list, relation_list, entity_id, relation_id)

def graph_size(kb_index: KBIndex) -> tuple[int, int]:
    return len(kb_index.entity_id), len(kb_index.relation_id)

def read_data(filename: str, kb_index: KBIndex, with_label: bool=False) -> Union[tuple[list[int], list[int], list[int]], tuple[list[int], list[int], list[int], list[int]]]:
    heads, relations, tails = [], [], []
    labels = []
    skipped_count = 0
    
    with open(filename) as f:
        for line in f:
            parts = line.strip().split('\t')
            h, r, t = parts[:3]
            
            # Check if entity and relation exist in kb_index
            if h not in kb_index.entity_id or r not in kb_index.relation_id or t not in kb_index.entity_id:
                skipped_count += 1
                continue
            
            # All entities and relations are valid, add to lists
            heads.append(kb_index.entity_id[h])
            relations.append(kb_index.relation_id[r])
            tails.append(kb_index.entity_id[t])

            if with_label and len(parts) > 3:
                labels.append(int(parts[3]))
    
    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} triples with entities/relations not in kb_index from {filename}")
    
    if with_label:
        return heads, relations, tails, labels
    else:
        return heads, relations, tails