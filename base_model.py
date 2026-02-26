import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Adam
from typing import Tuple, Optional
import logging
import os

import config

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
        self.set_device(use_gpu)

        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task, 'components')
        os.makedirs(self.task_dir, exist_ok=True)

    def set_device(self, use_gpu: bool = None) -> None:
        """Set runtime device for this BaseModel instance."""
        if use_gpu is None:
            # use device selected by config module
            try:
                self.device = config.device
            except AttributeError:
                self.device = torch.device('cpu')
        elif use_gpu:
            if torch.cuda.is_available():
                gpu_id = config.select_gpu()
                if gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    self.device = torch.device('cuda')
            else:
                logging.warning('Requested GPU but CUDA is not available. Falling back to CPU.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def load(self, model_path) -> None:
        pass

    def train(self, train_data, corrupter, tester,
              use_early_stopping=False, patience=10, optimizer_name='Adam',
              use_gpu: bool = None, is_save_model: bool = True) -> Tuple[float, str]:
        pass

    def save(self, save_path: Optional[str] = None) -> str:
        # Allow caller to override path at save time
        if save_path is not None:
            self.model_path = save_path

        if self.model_path is None:
            raise ValueError("Model path is not set. Cannot save model.")

        # Ensure destination directory exists
        dir_name = os.path.dirname(self.model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            torch.save(self.model.state_dict(), self.model_path)
        except Exception as e:
            logging.error(f"Error saving model: {e}")
        return self.model_path

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def constraint(self) -> None:
        self.model.constraint()

    def is_trained_or_loaded(self) -> bool:
        return self.model is not None

    def ensure_optimizer(self) -> None:
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.model.parameters(), weight_decay=self.weight_decay)
    
    def is_distance_based(self) -> bool:
        """Check if model is distance-based (lower score is better)."""
        return self.model_type in ['TransE', 'TransD']
    
    def get_score(self, head, relation, tail) -> torch.Tensor:
        return self.model.score(head, relation, tail)
    
    def get_prob_logit(self, head, relation, tail) -> torch.Tensor:
        return self.model.prob_logit(head, relation, tail)
