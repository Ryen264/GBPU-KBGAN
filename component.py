import logging
from typing import Generator, Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad

import config
import metrics
from datasets import (
    batch_by_num,
    batch_by_size,
    convert_data_to_no_label,
)
from models import ComplEx, DistMult, TransD, TransE
import loss

FILTER_RANKING_PENALTY = 1e30
OPTIMIZER_MAP = {
    "Adam": Adam,
    "SGD": SGD,
    "AdamW": AdamW,
    "RMSprop": RMSprop,
    "Adagrad": Adagrad,
}


def _join_rank_class_metrics(rank_value: float, class_value: float, class_rank_balance: float) -> float:
    """Blend rank and classification metric values using class_rank_balance in [0, 1]."""
    if not (0.0 <= class_rank_balance <= 1.0):
        raise ValueError("class_rank_balance must be in [0, 1].")
    return (1.0 - class_rank_balance) * rank_value + class_rank_balance * class_value


class Component:
    def __init__(self, role: str, model_type: str, n_entity: int, n_relation: int):
        """
        role = ["discriminator", "generator"]
        model_type = ["TransE", "TransD", "DistMult", "ComplEx"]
        """
        if role in ["discriminator", "generator"]:
            print(f"Initialized a new component with role {role}.")
            self.role = role
        else:
            raise ValueError("Input role should be in list [\"discriminator\", \"generator\"]!")

        if model_type in ["TransE", "TransD", "DistMult", "ComplEx"]:
            print(f"Initialized component: {model_type} model.")
            self.model_type = model_type
        else:
            raise ValueError("Input model type should be in list [\"TransE\", \"TransD\", \"DistMult\", \"ComplEx\"]!")

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.use_relation_attention = (self.role == "discriminator")
        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)

        self.model_path = self.model.model_path
        self.classification_threshold = None
        self.best_threshold_perf = {}
        self.relation_thresholds = {}
        self.global_threshold = None
        print(
            f"Initialized component successfully: {self.model_type} model with role {self.role}, "
            f"n_entity={self.n_entity}, n_relation={self.n_relation}."
        )

    def load(self, model_path: str) -> None:
        if self.n_entity is None or self.n_relation is None:
            raise ValueError("Component must be fitted before being loaded!")

        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation, use_relation_attention=self.use_relation_attention)
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
        self.classification_threshold = None
        self.global_threshold = None
        self.relation_thresholds = {}
        self.best_threshold_perf = {}

        if isinstance(checkpoint, dict) and (
            "state_dict" in checkpoint or "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
            self.classification_threshold = checkpoint.get("classification_threshold")
            self.global_threshold = checkpoint.get("global_threshold")
            self.relation_thresholds = checkpoint.get("relation_thresholds", {}) or {}
            self.best_threshold_perf = checkpoint.get("best_threshold_perf", {}) or {}
        else:
            state_dict = checkpoint

        model_state_keys = set(self.model.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        has_attention = any(key.startswith("relation_attention.") for key in model_state_keys)
        checkpoint_has_attention = any(key.startswith("relation_attention.") for key in checkpoint_keys)
        strict = has_attention == checkpoint_has_attention
        self.model.model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded component successfully by: {model_path}")

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = self.model.model_path

        checkpoint = {
            "state_dict": self.model.model.state_dict(),
            "classification_threshold": self.classification_threshold,
            "global_threshold": self.global_threshold,
            "relation_thresholds": self.relation_thresholds,
            "best_threshold_perf": self.best_threshold_perf,
            "model_type": self.model_type,
            "role": self.role,
        }
        torch.save(checkpoint, model_path)
        print(f"Saved component successfully by: {model_path}")

    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_var = Variable(head.to(config.device))
        relation_var = Variable(relation.to(config.device))
        tail_var = Variable(tail.to(config.device))
        return self.model.score(head_var, relation_var, tail_var)

    def dist(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_var = Variable(head.to(config.device))
        relation_var = Variable(relation.to(config.device))
        tail_var = Variable(tail.to(config.device))
        return self.model.dist(head_var, relation_var, tail_var)

    def embed(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Return entity embeddings for input entity ids across supported model types."""
        model_module = self.model.model
        if hasattr(model_module, "entity_embed"):
            return model_module.entity_embed(entity_ids)
        if hasattr(model_module, "entity_re_embed") and hasattr(model_module, "entity_im_embed"):
            return torch.cat(
                [
                    model_module.entity_re_embed(entity_ids),
                    model_module.entity_im_embed(entity_ids),
                ],
                dim=-1,
            )
        raise AttributeError("Model does not expose supported entity embedding layers.")

    def relation_embed(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Return relation embeddings for input relation ids across supported model types."""
        model_module = self.model.model
        if hasattr(model_module, "relation_embed"):
            return model_module.relation_embed(relation_ids)
        if hasattr(model_module, "relation_re_embed") and hasattr(model_module, "relation_im_embed"):
            return torch.cat(
                [
                    model_module.relation_re_embed(relation_ids),
                    model_module.relation_im_embed(relation_ids),
                ],
                dim=-1,
            )
        raise AttributeError("Model does not expose supported relation embedding layers.")

    def relation_attention(self, relation_ids: torch.Tensor) -> torch.Tensor:
        model_module = self.model.model
        if hasattr(model_module, "relation_attention") and model_module.relation_attention is not None:
            return model_module.relation_attention(relation_ids)
        raise AttributeError("Model does not expose relation attention weights.")

    def _directau_align_params(self) -> tuple[str, float]:
        kbgan_cfg = config._config["KBGAN"]
        align_op = kbgan_cfg.get("emb_align_op", "add")
        align_balance = kbgan_cfg.get("emb_align_balance", 0.7)
        if align_op not in ["add", "mul"]:
            raise ValueError("emb_align_op must be one of ['add', 'mul']")
        if not (0.0 <= align_balance <= 1.0):
            raise ValueError("emb_align_balance must be in [0, 1]")
        return align_op, align_balance

    def _distance_score(self, head_ids: torch.Tensor, relation_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        align_op, align_balance = self._directau_align_params()
        return loss.align_distance_sq(
            head_ids=head_ids,
            relation_ids=relation_ids,
            tail_ids=tail_ids,
            entity_emb=self.embed,
            relation_emb=self.relation_embed,
            attention_emb=self.relation_attention if self.use_relation_attention else None,
            align_balance=align_balance,
            align_op=align_op,
        )

    def train(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        train_data: tuple,
        valid_data_w_label: tuple,
        class_rank_balance: float = 1.0,
        early_stop_patience: int = -1,
        rank_optimizing_metric: str = "mrr",
        rank_filt: bool = True,
        rank_k_list: list = [1, 3, 10],
        class_optimizing_metric: str = "accuracy",
    ) -> float:
        config.overwrite_config_with_args(["--log.prefix=" + self.model_type + "_"])
        config.logger_init()

        if class_rank_balance < 0.0 or class_rank_balance > 1.0:
            raise ValueError("class_rank_balance must be in [0, 1].")

        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        head, relation, tail = train_data
        n_train = len(head)

        # Stage-1 DirectAU-style pretraining hyperparameters.
        kbgan_cfg = config._config["KBGAN"]
        uni_scale = kbgan_cfg.get("emb_uniform_scale", 2.0)
        align_gamma = kbgan_cfg.get("true_align_gamma", kbgan_cfg.get("emb_loss_gamma", 1.0))
        align_op = kbgan_cfg.get("emb_align_op", "add")
        align_balance = kbgan_cfg.get("emb_align_balance", 0.7)
        entity_uniform_max_ids = kbgan_cfg.get("entity_uniform_max_ids", None)

        if align_op not in ["add", "mul"]:
            raise ValueError("emb_align_op must be one of ['add', 'mul']")
        if not (0.0 <= align_balance <= 1.0):
            raise ValueError("emb_align_balance must be in [0, 1]")
        if entity_uniform_max_ids is not None and entity_uniform_max_ids < 2:
            raise ValueError("entity_uniform_max_ids must be >= 2 or None")

        valid_data_no_label = convert_data_to_no_label(valid_data_w_label)
        rank_metrics = lambda: self.evaluate_on_ranking(
            valid_data_no_label, heads, tails, filt=rank_filt, k_list=rank_k_list
        )

        class_metrics = lambda: self.evaluate_on_classification(
            valid_data_w_label, optimizing_metric=class_optimizing_metric, is_threshold_tunning=True
        )
        tester = lambda: _join_rank_class_metrics(
            rank_value=rank_metrics()[rank_optimizing_metric],
            class_value=class_metrics()[class_optimizing_metric],
            class_rank_balance=class_rank_balance,
        )

        print(f"Start training component: {self.model_type} model with role {self.role}...")

        n_epoch = getattr(self.model, "n_epoch", 100)
        n_batch = getattr(self.model, "n_batch", 100)
        epoch_per_test = getattr(self.model, "epoch_per_test", 10)
        best_perf = 0.0
        best_epoch = -1
        patience_counter = 0

        for epoch in range(n_epoch):
            epoch_loss = 0.0
            rand_idx = torch.randperm(n_train)
            head_epoch = head[rand_idx]
            relation_epoch = relation[rand_idx]
            tail_epoch = tail[rand_idx]

            for h, r, t in batch_by_num(n_batch, head_epoch, relation_epoch, tail_epoch, n_sample=n_train):
                batch_size = h.size(0)
                h_device = h.to(config.device)
                r_device = r.to(config.device)
                t_device = t.to(config.device)

                entity_ids_batch = torch.unique(torch.cat((h_device.reshape(-1), t_device.reshape(-1)), dim=0))
                if entity_uniform_max_ids is not None and entity_ids_batch.numel() > entity_uniform_max_ids:
                    sample_idx = torch.randperm(entity_ids_batch.numel(), device=entity_ids_batch.device)[:entity_uniform_max_ids]
                    entity_ids_batch = entity_ids_batch[sample_idx]

                # Stage-1 objective: mean align distance + gamma * batch entity uniformity.
                align_dist_sq = loss.align_distance_sq(
                    head_ids=h_device,
                    relation_ids=r_device,
                    tail_ids=t_device,
                    entity_emb=self.embed,
                    relation_emb=self.relation_embed,
                    attention_emb=self.relation_attention if self.use_relation_attention else None,
                    align_balance=align_balance,
                    align_op=align_op,
                )
                align_loss_batch = align_dist_sq.mean()
                uni_loss_batch = loss.uniform_loss(
                    ids=entity_ids_batch,
                    emb=self.embed,
                    scale=uni_scale,
                )
                batch_loss = align_loss_batch + align_gamma * uni_loss_batch

                self.opt_zero_grad()
                batch_loss.backward()
                self.opt_step()
                epoch_loss += batch_loss.detach().item() * batch_size

            logging.info("Epoch %d/%d, Loss=%f", epoch + 1, n_epoch, epoch_loss / n_train)
            if (n_epoch >= epoch_per_test) and ((epoch + 1) % epoch_per_test == 0):
                test_perf = tester()
                if test_perf > best_perf:
                    self.save()
                    best_perf = test_perf
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    logging.info(
                        "Early stopping triggered at epoch %d (patience=%d)",
                        epoch + 1,
                        early_stop_patience,
                    )
                    break

        if best_epoch > 0:
            self.load(self.model_path)
        print(f"Trained component successfully: {self.model_type} model.")
        return best_perf, best_epoch

    def opt_zero_grad(self) -> None:
        self.model.ensure_optimizer()
        self.model.opt.zero_grad()

    def opt_step(self) -> None:
        self.model.opt.step()
        self.model.constraint()

    def set_optimizer(self, optimizer_name: str, lr: float, weight_decay: float = 0.0) -> None:
        opt_cls = OPTIMIZER_MAP.get(optimizer_name, Adam)
        try:
            self.model.opt = opt_cls(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        except (AttributeError, TypeError):
            pass

    def generator_step(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        n_sample: int = 1,
        temperature: float = 1.0,
        train: bool = True,
        sampling_strategy: str = "multinomial",
    ) -> Generator[torch.Tensor, torch.Tensor, None]:
        """
        Generator step: sample fake triples and update with REINFORCE.
        n_sample controls number of sampled fake triples per input triple.
        sampling_strategy: 'multinomial' or 'topk'.
        """
        if self.role != "generator":
            raise ValueError("This component is not a generator!")
        if sampling_strategy not in ["multinomial", "topk"]:
            raise ValueError("sampling_strategy must be one of ['multinomial', 'topk']")

        # Forward pass: generate samples
        n, m = tail.size()
        if n_sample > m:
            raise ValueError(f"n_sample ({n_sample}) cannot be larger than candidate pool size ({m}).")

        relation_var = Variable(relation.to(config.device))
        head_var = Variable(head.to(config.device))
        tail_var = Variable(tail.to(config.device))

        logits = self.model.prob_logit(head_var, relation_var, tail_var) / temperature
        probs = nnf.softmax(logits, dim=-1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        if sampling_strategy == "topk":
            sample_idx = torch.topk(probs, k=n_sample, dim=-1, largest=True, sorted=False).indices
        else:
            sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_heads = head[row_idx, sample_idx.data.cpu()]
        sample_tails = tail[row_idx, sample_idx.data.cpu()]

        # Yield samples to get rewards from discriminator
        rewards = yield sample_heads, sample_tails

        # Backward pass: update generator with REINFORCE
        if train:
            self.opt_zero_grad()
            log_probs = nnf.log_softmax(logits, dim=-1)
            # Move indices to device for proper indexing
            row_idx_device = row_idx.to(config.device)
            sample_idx_device = sample_idx.to(config.device)
            # Ensure rewards are a tensor and on device
            rewards_tensor = (
                rewards
                if isinstance(rewards, torch.Tensor)
                else torch.tensor(rewards, device=config.device, dtype=torch.float32)
            )
            if rewards_tensor.device != config.device:
                rewards_tensor = rewards_tensor.to(config.device)
            # Compute REINFORCE loss: -sum(rewards * log_probs)
            reinforce_loss = -torch.sum(rewards_tensor * log_probs[row_idx_device, sample_idx_device])
            reinforce_loss.backward()
            self.opt_step()
        yield None

    def discriminator_step(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        head_fake: torch.Tensor,
        tail_fake: torch.Tensor,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Discriminator step: distinguish real from fake triples
        """
        if self.role != "discriminator":
            raise ValueError("This component is not a discriminator!")

        # Forward pass: compute losses and scores
        head_var = Variable(head.to(config.device))
        relation_var = Variable(relation.to(config.device))
        tail_var = Variable(tail.to(config.device))
        head_fake_var = Variable(head_fake.to(config.device))
        tail_fake_var = Variable(tail_fake.to(config.device))

        if head_fake_var.dim() == relation_var.dim() + 1:
            relation_fake_var = relation_var.unsqueeze(1).expand_as(head_fake_var)
            d_good = self.model.dist(head_var, relation_var, tail_var).unsqueeze(1).expand_as(head_fake_var)
        else:
            relation_fake_var = relation_var
            d_good = self.model.dist(head_var, relation_var, tail_var)

        d_bad = self.model.dist(head_fake_var, relation_fake_var, tail_fake_var)
        pair_loss = nnf.relu(d_good - d_bad + self.model.margin)
        fake_scores = self.model.score(head_fake_var, relation_fake_var, tail_fake_var)

        # In training mode, return differentiable pair_loss so caller can build a joint objective.
        # In evaluation mode, detach to avoid building autograd graph.
        pair_loss_out = pair_loss if train else pair_loss.detach()
        return pair_loss_out, -fake_scores.detach(), d_good.detach().max().item(), d_bad.detach().min().item()

    def evaluate_on_ranking(
        self,
        test_data: tuple,
        heads: torch.Tensor,
        tails: torch.Tensor,
        filt: bool = True,
        k_list: list = [1, 3, 10],
    ) -> dict:
        mr_total = mrr_total = 0.0
        hits_total = [0] * len(k_list)
        count = 0
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(self.model.test_batch_size, *test_data):
                batch_head = torch.LongTensor(batch_head)
                batch_relation = torch.LongTensor(batch_relation)
                batch_tail = torch.LongTensor(batch_tail)
                batch_size = batch_head.size(0)

                all_var = (
                    torch.arange(0, self.n_entity)
                    .unsqueeze(0)
                    .expand(batch_size, self.n_entity)
                    .long()
                    .to(config.device)
                )
                head_var = batch_head.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                relation_var = (
                    batch_relation.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)
                )
                tail_var = batch_tail.unsqueeze(1).expand(batch_size, self.n_entity).to(config.device)

                # Stage-3 inference: rank by DirectAU distance d = ||q - e||^2 (ascending).
                batch_head_scores = self._distance_score(all_var, relation_var, tail_var).detach()
                batch_tail_scores = self._distance_score(head_var, relation_var, all_var).detach()

                for head, relation, tail, head_scores, tail_scores in zip(
                    batch_head, batch_relation, batch_tail, batch_head_scores, batch_tail_scores
                ):
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

                    head_metrics = metrics.ranking_metrics(scores=head_scores, target=head_id, k_list=k_list)
                    tail_metrics = metrics.ranking_metrics(scores=tail_scores, target=tail_id, k_list=k_list)

                    head_mr, head_mrr, head_hits = (
                        head_metrics["mr"],
                        head_metrics["mrr"],
                        head_metrics["hits"],
                    )
                    tail_mr, tail_mrr, tail_hits = (
                        tail_metrics["mr"],
                        tail_metrics["mrr"],
                        tail_metrics["hits"],
                    )

                    mr_total += head_mr + tail_mr
                    mrr_total += head_mrr + tail_mrr
                    hits_total = [
                        (hits_total[i] + head_hits[i] + tail_hits[i]) for i in range(len(k_list))
                    ]
                    count += 2

        mr_rate = mr_total / count
        mrr_rate = mrr_total / count
        hits_rate = [hit_total / count for hit_total in hits_total]

        ranking_metrics = {}
        ranking_metrics["mr"] = mr_rate
        ranking_metrics["mrr"] = mrr_rate
        for i in range(len(k_list)):
            ranking_metrics[f"hit@{k_list[i]}"] = hits_rate[i]

        # Format metrics for cleaner output
        parts = []
        label_map = {"mr": "MR", "mrr": "MRR"}
        for k, v in ranking_metrics.items():
            label = label_map.get(k, k.replace("hit@", "Hit@"))
            parts.append(f"{label}: {v:.4f}")
        ranking_metrics_str = f"Ranking metrics: {', '.join(parts)}\n"
        logging.info(ranking_metrics_str)
        return ranking_metrics

    def find_optimal_threshold(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 100,
        optimizing_metric: str = "accuracy",
    ) -> Tuple[float, float]:
        """
        Find the optimal threshold for triple classification from distance scores.

        Unified decision rule in this codebase:
        distance <= threshold => positive (label=1).

        Args:
            distances: Distance values for validation samples.
            labels: Ground truth labels for validation samples.
            n_thresholds: Number of threshold values to try

        Returns:
            (Optimal threshold value, best validation score for optimizing_metric)
        """
        if distances.size == 0:
            raise ValueError("No distances provided to find_optimal_threshold.")
        min_score = float(distances.min())
        max_score = float(distances.max())
        threshold_values = np.linspace(min_score, max_score, n_thresholds)

        best_val = -float("inf")
        best_threshold = 0.0

        for threshold in threshold_values:
            predictions = np.where(distances <= threshold, 1, 0).tolist()
            scores_for_auc = (-distances).tolist()
            candidate_metrics = metrics.classification_metrics(
                predictions, labels, scores=scores_for_auc
            )
            candidate_val = candidate_metrics.get(optimizing_metric, 0.0)

            if candidate_val > best_val:
                best_val = candidate_val
                best_threshold = threshold
        return best_threshold, best_val

    def evaluate_on_classification(
        self,
        test_data_w_label: tuple,
        optimizing_metric: str = "accuracy",
        is_threshold_tunning: bool = False,
        external_threshold: float = None,
    ) -> dict:
        if len(test_data_w_label) < 4:
            raise ValueError(
                "For classification metrics, test_data_w_label must include labels as the 4th element "
                "(heads, relations, tails, labels)."
            )

        heads_list, relations_list, tails_list, labels = test_data_w_label
        distances_list = []
        relation_ids_list = []
        true_labels = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(
                self.model.test_batch_size,
                heads_list,
                relations_list,
                tails_list,
                labels,
            ):
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)
                batch_distances = self._distance_score(head_var, relation_var, tail_var).detach().cpu().tolist()
                distances_list.extend([float(s) for s in batch_distances])
                relation_ids_list.extend([int(x) for x in batch_relation])
                true_labels.extend([int(x) for x in batch_label])
        if len(distances_list) == 0:
            raise ValueError("No samples found in test_data for classification evaluation.")

        distances_array = np.array(distances_list, dtype=np.float64)
        relation_ids_array = np.array(relation_ids_list, dtype=np.int64)
        true_labels_array = np.array(true_labels, dtype=np.int64)

        threshold = None
        if is_threshold_tunning:
            n_thresholds = 100
            relation_thresholds = {}
            for relation_id in np.unique(relation_ids_array):
                rel_mask = relation_ids_array == relation_id
                if not np.any(rel_mask):
                    continue
                relation_thresholds[int(relation_id)], _ = self.find_optimal_threshold(
                    distances=distances_array[rel_mask],
                    labels=true_labels_array[rel_mask],
                    n_thresholds=n_thresholds,
                    optimizing_metric=optimizing_metric,
                )

            candidate_threshold, _ = self.find_optimal_threshold(
                distances=distances_array,
                labels=true_labels_array,
                n_thresholds=n_thresholds,
                optimizing_metric=optimizing_metric,
            )
            candidate_predictions = []
            for dist, relation_id in zip(distances_array.tolist(), relation_ids_array.tolist()):
                relation_threshold = relation_thresholds.get(int(relation_id), candidate_threshold)
                candidate_predictions.append(1 if dist <= relation_threshold else 0)
            candidate_scores_for_auc = (-distances_array).tolist()
            candidate_metrics = metrics.classification_metrics(
                candidate_predictions,
                true_labels,
                scores=candidate_scores_for_auc,
            )
            candidate_val = candidate_metrics.get(optimizing_metric, 0.0)

            best_so_far = self.best_threshold_perf.get(optimizing_metric, -float("inf"))
            if self.classification_threshold is None or candidate_val > best_so_far:
                self.classification_threshold = candidate_threshold
                self.global_threshold = candidate_threshold
                self.relation_thresholds = relation_thresholds
                self.best_threshold_perf[optimizing_metric] = candidate_val
                logging.info(
                    f"Updated thresholds ({optimizing_metric}={candidate_val:.4f}): "
                    f"distance <= threshold => positive, relation_thresholds={len(self.relation_thresholds)}, "
                    f"global_threshold={self.global_threshold:.4f}."
                )
            else:
                logging.info(
                    f"Kept previous thresholds ({optimizing_metric}={best_so_far:.4f}). "
                    f"Current tuned threshold score is {candidate_val:.4f}."
                )
            threshold = self.classification_threshold
            logging.info(
                f"Using relation-specific thresholds with global fallback={self.classification_threshold:.4f} "
                f"(searched {n_thresholds} thresholds per relation/global)."
            )
        else:
            if external_threshold is not None:
                threshold = float(external_threshold)
            else:
                threshold = self.global_threshold if self.global_threshold is not None else self.classification_threshold

        if threshold is None:
            raise ValueError(
                "Classification threshold is None. Provide external_threshold or run with "
                "is_threshold_tunning=True first. Decision rule is fixed: "
                "distance <= threshold => positive."
            )

        predictions = []
        if external_threshold is not None:
            predictions = np.where(distances_array <= threshold, 1, 0).tolist()
        else:
            for dist, relation_id in zip(distances_array.tolist(), relation_ids_array.tolist()):
                relation_threshold = self.relation_thresholds.get(int(relation_id), threshold)
                predictions.append(1 if dist <= relation_threshold else 0)

        # Convert distances so that larger values indicate positive class for AUC.
        scores_for_auc = (-distances_array).tolist()

        classification_metrics = metrics.classification_metrics(
            predictions, true_labels, scores=scores_for_auc
        )
        # Format metrics for cleaner output
        parts = []
        label_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "pr_auc": "PR AUC",
            "roc_auc": "ROC AUC",
        }
        for k, v in classification_metrics.items():
            label = label_map.get(k, k)
            parts.append(f"{label}: {v:.4f}")
        classification_metrics_str = f"Classification metrics: {', '.join(parts)}"

        threshold_source = None
        if external_threshold is not None:
            threshold_source = "External threshold"
        else:
            threshold_source = "Relation threshold + global fallback"

        logging.info(classification_metrics_str)
        logging.info(
            f"[Classification] threshold_source={threshold_source}, "
            f"threshold={threshold:.6f}, decision_rule='distance<=threshold=>positive'"
        )
        return classification_metrics
