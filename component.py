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
    BernCorrupter,
    BernCorrupterMulti,
    batch_by_size,
    convert_data_to_no_label,
)
from models import ComplEx, DistMult, TransD, TransE

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
        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation)

        self.model_path = self.model.model_path
        self.classification_threshold = None
        self.best_threshold_perf = {}
        print(
            f"Initialized component successfully: {self.model_type} model with role {self.role}, "
            f"n_entity={self.n_entity}, n_relation={self.n_relation}."
        )

    def load(self, model_path: str) -> None:
        if self.n_entity is None or self.n_relation is None:
            raise ValueError("Component must be fitted before being loaded!")

        if self.model_type == "TransE":
            self.model = TransE(self.n_entity, self.n_relation)
        elif self.model_type == "TransD":
            self.model = TransD(self.n_entity, self.n_relation)
        elif self.model_type == "DistMult":
            self.model = DistMult(self.n_entity, self.n_relation)
        elif self.model_type == "ComplEx":
            self.model = ComplEx(self.n_entity, self.n_relation)
        self.model.load(model_path)
        print(f"Loaded component successfully by: {model_path}")

    def save(self, model_path: str = None):
        if model_path is None:
            model_path = self.model.model_path

        self.model.save(model_path)
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

        # Convert to plain lists for BernCorrupter so dict keys use value-based hashing
        train_data_list = [d.tolist() if isinstance(d, torch.Tensor) else d for d in train_data]
        if self.model_type in ["TransE", "TransD"]:
            corrupter = BernCorrupter(train_data_list, self.n_entity, self.n_relation)
        elif self.model_type in ["DistMult", "ComplEx"]:
            corrupter = BernCorrupterMulti(train_data_list, self.n_entity, self.n_relation, self.model.n_sample)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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
        best_perf, best_epoch = self.model.train(
            train_data, corrupter, tester, early_stop_patience=early_stop_patience
        )
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

                batch_head_scores = self.model.score(all_var, relation_var, tail_var).detach()
                batch_tail_scores = self.model.score(head_var, relation_var, all_var).detach()

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
        valid_data: tuple,
        labels: list,
        n_thresholds: int = 100,
        optimizing_metric: str = "accuracy",
    ) -> Tuple[float, float]:
        """
        Find the optimal threshold for triple classification using validation data.

        Unified decision rule in this codebase:
        score < threshold => positive (label=1).

        Args:
            valid_data: Tuple of (heads, relations, tails)
            labels: Ground truth labels for validation data
            n_thresholds: Number of threshold values to try

        Returns:
            (Optimal threshold value, best validation score for optimizing_metric)
        """
        heads, relations, tails = valid_data

        # Compute scores for all validation samples (batched for efficiency)
        scores_list = []
        with torch.no_grad():
            for batch_head, batch_relation, batch_tail in batch_by_size(
                self.model.test_batch_size, heads, relations, tails
            ):
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)

                batch_scores = self.model.score(head_var, relation_var, tail_var)
                batch_scores = batch_scores.detach().cpu().numpy()
                scores_list.extend(batch_scores.tolist())

        # Try different threshold values
        scores_array = np.array(scores_list)
        min_score = float(scores_array.min())
        max_score = float(scores_array.max())
        threshold_values = np.linspace(min_score, max_score, n_thresholds)

        best_val = -float("inf")
        best_threshold = 0.0

        for threshold in threshold_values:
            predictions = np.where(scores_array < threshold, 1, 0).tolist()
            scores_for_auc = (-scores_array).tolist()
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
        scores_list = []
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
                batch_scores = self.model.score(head_var, relation_var, tail_var).detach().cpu().tolist()
                scores_list.extend([float(s) for s in batch_scores])
                true_labels.extend([int(x) for x in batch_label])
        if len(scores_list) == 0:
            raise ValueError("No samples found in test_data for classification evaluation.")

        threshold = None
        if is_threshold_tunning:
            n_thresholds = 100
            candidate_threshold, candidate_val = self.find_optimal_threshold(
                valid_data=(heads_list, relations_list, tails_list),
                labels=true_labels,
                n_thresholds=n_thresholds,
                optimizing_metric=optimizing_metric,
            )
            best_so_far = self.best_threshold_perf.get(optimizing_metric, -float("inf"))
            if self.classification_threshold is None or candidate_val > best_so_far:
                self.classification_threshold = candidate_threshold
                self.best_threshold_perf[optimizing_metric] = candidate_val
                logging.info(
                    f"Updated best classification threshold ({optimizing_metric}={candidate_val:.4f}): "
                    f"score < {self.classification_threshold:.4f} => positive."
                )
            else:
                logging.info(
                    f"Kept previous best classification threshold ({optimizing_metric}={best_so_far:.4f}). "
                    f"Current tuned threshold score is {candidate_val:.4f}."
                )
            threshold = self.classification_threshold
            logging.info(
                f"Using classification threshold: score < {self.classification_threshold:.4f} => positive "
                f"(searched {n_thresholds} thresholds)."
            )
        else:
            if external_threshold is not None:
                threshold = external_threshold
            else:
                threshold = self.classification_threshold

        if threshold is None:
            raise ValueError(
                "Classification threshold is None. Provide external_threshold or run with "
                "is_threshold_tunning=True first. Decision rule is fixed: "
                "score < threshold => positive."
            )

        # Fixed rule in this codebase: lower score means more likely positive.
        scores_array = np.array(scores_list)
        predictions = np.where(scores_array < threshold, 1, 0).tolist()

        # Convert scores so that larger values indicate positive class for AUC
        scores_for_auc = [-s for s in scores_list]

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
        classification_metrics_str = f"Classification metrics: {', '.join(parts)}\n"

        threshold_source = None
        if external_threshold is not None:
            threshold_source = "External threshold"
        else:
            threshold_source = "Internal threshold"

        logging.info(classification_metrics_str)
        logging.info(
            f"[Classification] threshold_source={threshold_source}, "
            f"threshold={threshold:.6f}, decision_rule='score<threshold=>positive'"
        )
        return classification_metrics
