import os
import logging
import torch
from typing import Generator, Tuple
import numpy as np
from datetime import datetime

from datasets import batch_by_num, batch_by_size, BernCorrupterMulti
from component import Component
import loss
import config

def _join_rank_class_metrics(rank_value: float, class_value: float, class_rank_balance: float) -> float:
    """Blend rank and classification metric values using class_rank_balance in [0, 1]."""
    if not (0.0 <= class_rank_balance <= 1.0):
        raise ValueError("class_rank_balance must be in [0, 1].")
    return (1.0 - class_rank_balance) * rank_value + class_rank_balance * class_value

def _safe_stats(name: str, values: list) -> dict:
    """Return robust distribution stats for a list of float scores."""
    if len(values) == 0:
        return {
            "name": name,
            "n_sample": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "percentiles": {}
        }

    arr = np.asarray(values, dtype=np.float64)
    percentile_points = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    return {
        "name": name,
        "n_sample": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "percentiles": {p: float(np.percentile(arr, p)) for p in percentile_points},
    }

class KBGAN():
    def __init__(self, discriminator_type: str, generator_type: str, n_entity: int, n_relation: int):
        """
        discriminator_type = ["TransE", "TransD"]
        generator_type = ["DistMult", "ComplEx"]
        """
        self.discriminator_type = discriminator_type
        self.generator_type = generator_type
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.discriminator = Component(role="discriminator", model_type=discriminator_type,
                                       n_entity=n_entity, n_relation=n_relation)
        self.generator = Component(role="generator", model_type=generator_type,
                                   n_entity=n_entity, n_relation=n_relation)

        self.discriminator_path = self.discriminator.model_path
        self.generator_path = self.generator.model_path

        self.dataset = config._config.dataset
        self.task = config._config.task
        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
        os.makedirs(self.task_dir, exist_ok=True)

        self.model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        self.kbgan_path = os.path.join(self.task_dir, self.model_name)
        self.optimal_threshold = None

    def load_discriminator(self, filepath: str) -> None:
        self.discriminator.load(filepath)
        print(f"Loaded discriminator successfully by path: {filepath}")

    def load_generator(self, filepath: str) -> None:
        self.generator.load(filepath)
        print(f"Loaded generator successfully by path: {filepath}")

    def load_kbgan(self, filepath: str) -> None:      
        self.discriminator.load(filepath)
        print(f"Loaded KBGAN (discriminator) successfully by: {filepath}")

    def save_kbgan(self, filepath: str=None) -> None:
        if filepath is None:
            filepath = self.kbgan_path
        self.discriminator.save(filepath)
        print(f"Saved KBGAN (discriminator) successfully to: {filepath}")
        
    def train_components(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                        class_rank_balance: float=1.0, early_stop_patience: int=-1,
                        rank_optimizing_metric: str='mrr', rank_filt: bool=True, rank_k_list: list=[1, 3, 10],
                        class_optimizing_metric: str='accuracy') -> Tuple[float, float]:
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        best_perf_d, best_epoch_d = self.discriminator.train(heads, tails, train_data, valid_data_w_label,
                                                            class_rank_balance=class_rank_balance,
                                                            early_stop_patience=early_stop_patience,
                                                            rank_optimizing_metric=rank_optimizing_metric,
                                                            rank_filt=rank_filt, rank_k_list=rank_k_list,
                                                            class_optimizing_metric=class_optimizing_metric)
        print(f"Trained {self.discriminator_type} discriminator successfully with performance: {best_perf_d}, epoch: {best_epoch_d}")

        best_perf_g, best_epoch_g = self.generator.train(heads, tails, train_data, valid_data_w_label,
                                                        class_rank_balance=class_rank_balance,
                                                        early_stop_patience=early_stop_patience,
                                                        rank_optimizing_metric=rank_optimizing_metric,
                                                        rank_filt=rank_filt, rank_k_list=rank_k_list,
                                                        class_optimizing_metric=class_optimizing_metric)
        print(f"Trained {self.generator_type} generator successfully with performance: {best_perf_g}, epoch: {best_epoch_g}")
        return best_perf_d, best_perf_g

    def generator_step(self, hs: torch.Tensor, rs: torch.Tensor, ts: torch.Tensor,
                       n_sample: int, temperature: float,
                       negative_sampling_strategy: str
        ) -> Tuple[Generator, torch.Tensor, torch.Tensor]:
        """
        KBGAN-level generator step: sample fake triples and return live generator for reward update.
        """
        gen_step = self.generator.generator_step(
            hs, rs, ts,
            n_sample=n_sample,
            temperature=temperature,
            train=True,
            sampling_strategy=negative_sampling_strategy,
        )
        head_smpl, tail_smpl = next(gen_step)
        head_smpl_device = head_smpl.to(config.device)
        tail_smpl_device = tail_smpl.to(config.device)
        return gen_step, head_smpl_device, tail_smpl_device

    def discriminator_step(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
                           emb_uniform_p: float, emb_uniform_scale: float,
                           true_align_gamma: float, fake_align_gamma: float,
                           emb_align_op: str, emb_align_balance: float,
                           head_fake: torch.Tensor=None, tail_fake: torch.Tensor=None,
                           return_fake_align: bool=False,
        ) -> torch.Tensor:
        """
        KBGAN-level discriminator objective: pairwise true-vs-fake loss + embedding regularization.
        """
        h_device, r_device, t_device = h.to(config.device), r.to(config.device), t.to(config.device)
        entity_ids_full = torch.arange(self.n_entity, device=config.device, dtype=torch.long)
        
        ent_uni_loss = loss.uniform_loss(
            ids=entity_ids_full,
            emb=self.discriminator.embed,
            scale=emb_uniform_scale
        )
        rel_uni_loss = loss.uniform_loss(
            ids=r_device,
            emb=self.discriminator.relation_embed,
            scale=emb_uniform_scale
        )
        true_ali_loss = loss.align_loss(
            head_ids=h_device,
            relation_ids=r_device,
            tail_ids=t_device,
            entity_emb=self.discriminator.embed,
            relation_emb=self.discriminator.relation_embed,
            align_balance=emb_align_balance,
            align_op=emb_align_op,
        )
        emb_reg_loss = true_align_gamma * true_ali_loss + emb_uniform_p * ent_uni_loss + (1.0 - emb_uniform_p) * rel_uni_loss

        if head_fake is None or tail_fake is None:
            return emb_reg_loss

        head_fake_device = head_fake.to(config.device)
        tail_fake_device = tail_fake.to(config.device)
        if head_fake_device.dim() == r_device.dim() + 1:
            relation_for_fake = r_device.unsqueeze(1).expand_as(head_fake_device)
        else:
            relation_for_fake = r_device

        fake_ali_loss = loss.align_loss(
            head_ids=head_fake_device,
            relation_ids=relation_for_fake,
            tail_ids=tail_fake_device,
            entity_emb=self.discriminator.embed,
            relation_emb=self.discriminator.relation_embed,
            align_balance=emb_align_balance,
            align_op=emb_align_op,
        )
        total_loss = emb_reg_loss - fake_align_gamma * fake_ali_loss
        if return_fake_align:
            return total_loss, fake_ali_loss.detach()
        return total_loss

    def _run_validation_epoch(self,
                              epoch: int,
                              n_epoch: int,
                              valid_data_w_label: tuple,
                              heads: torch.Tensor,
                              tails: torch.Tensor,
                              temperature: float,
                              do_class_task: bool,
                              class_rank_balance: float,
                              rank_optimizing_metric: str,
                              rank_filt: bool,
                              rank_k_list: list,
                              class_optimizing_metric: str,
                              class_use_maxgood_minbad_threshold: bool,
                              n_generated_valid_negative: int,
                              class_true_percentile: float,
                              class_fake_percentile: float,
                              class_true_fake_balance: float,
        ) -> float:
        # [VALID LOSS SNAPSHOT]
        if len(valid_data_w_label) >= 4:
            valid_head_all, valid_relation_all, valid_tail_all, valid_label_all = valid_data_w_label
            valid_labels_np = np.asarray(valid_label_all)
            valid_pos_mask = (valid_labels_np == 1)
            if np.any(valid_pos_mask):
                valid_head_eval = torch.LongTensor(np.asarray(valid_head_all)[valid_pos_mask].astype(np.int64))
                valid_relation_eval = torch.LongTensor(np.asarray(valid_relation_all)[valid_pos_mask].astype(np.int64))
                valid_tail_eval = torch.LongTensor(np.asarray(valid_tail_all)[valid_pos_mask].astype(np.int64))
            else:
                valid_head_eval, valid_relation_eval, valid_tail_eval = valid_data_w_label[:3]
        else:
            valid_head_eval, valid_relation_eval, valid_tail_eval = valid_data_w_label[:3]

        if not isinstance(valid_head_eval, torch.Tensor):
            valid_head_eval = torch.LongTensor(valid_head_eval)
        if not isinstance(valid_relation_eval, torch.Tensor):
            valid_relation_eval = torch.LongTensor(valid_relation_eval)
        if not isinstance(valid_tail_eval, torch.Tensor):
            valid_tail_eval = torch.LongTensor(valid_tail_eval)

        # [RANK TASK]
        valid_data_no_label = valid_data_w_label[:3]
        rank_metrics = self.discriminator.evaluate_on_ranking(
            valid_data_no_label,
            heads,
            tails,
            filt=rank_filt,
            k_list=rank_k_list,
        )

        # [CLASS TASK]
        if do_class_task:
            class_threshold = None
            is_threshold_tunning = False
            # [MAXGOOD MINBAD THRESHOLD]
            if class_use_maxgood_minbad_threshold:
                class_threshold = self._compute_midpoint_threshold_from_labeled_data(
                    valid_data_w_label,
                    n_generated_valid_negative=n_generated_valid_negative,
                    temperature=temperature,
                    true_percentile=class_true_percentile,
                    fake_percentile=class_fake_percentile,
                    true_fake_balance=class_true_fake_balance,
                )
                if class_threshold is not None:
                    self.optimal_threshold = class_threshold
                    logging.info(f"Using validation midpoint threshold: {class_threshold:.6f}")
                else:
                    # Fallback: tune threshold from validation metrics when midpoint cannot be computed.
                    is_threshold_tunning = True
                    logging.warning("Validation midpoint threshold unavailable. Falling back to threshold tuning on validation set.")
            else:
                # Default behavior when midpoint rule is disabled: tune threshold on validation set.
                is_threshold_tunning = True
            class_metrics = self.discriminator.evaluate_on_classification(
                valid_data_w_label,
                optimizing_metric=class_optimizing_metric,
                is_threshold_tunning=is_threshold_tunning,
                external_threshold=class_threshold,
            )
            # [JOINT METRIC]
            test_perf = (1.0 - class_rank_balance) * rank_metrics[rank_optimizing_metric] \
                        + class_rank_balance * class_metrics[class_optimizing_metric]
            
            log_msg = f"Valid epoch {epoch + 1}/{n_epoch}, perf={test_perf}"
            log_msg += f"\n\t{rank_optimizing_metric}={rank_metrics[rank_optimizing_metric]}, {class_optimizing_metric}={class_metrics[class_optimizing_metric]}"
        else:
            test_perf = rank_metrics[rank_optimizing_metric]
            log_msg = f"Valid epoch {epoch + 1}/{n_epoch}, perf={test_perf}"
        logging.info(log_msg)
        return test_perf

    def train_kbgan(self, heads: torch.Tensor, tails: torch.Tensor, train_data: tuple, valid_data_w_label: tuple,
                class_rank_balance: float=1.0,
                early_stop_patience: int=-1,
                temperature: float=1.0,
                n_sample: int=20,
                n_candidate: int=None,
                n_epoch: int=5000,
                n_batch: int=100,
                epoch_per_test: int=100,
                n_generated_valid_negative: int=0,
                negative_sampling_strategy: str='multinomial',
                emb_uniform_p: float=0.5,
                emb_uniform_scale: float=2.0,
                true_align_gamma: float=1.0,
                fake_align_gamma: float=1.0,
                emb_align_op: str='add',
                emb_align_balance: float=0.7,
                rank_optimizing_metric: str='mrr',
                rank_filt: bool=True,
                rank_k_list: list=[1, 3, 10],
                class_optimizing_metric: str='accuracy',
                class_use_maxgood_minbad_threshold: bool=True,
                class_true_percentile: float=90.0,
                class_fake_percentile: float=5.0,
                class_true_fake_balance: float=0.5,
                ) -> float:
        """
        class_rank_balance is a ratio in [0, 1]:
        - 0.0 => optimize ranking only
        - 1.0 => optimize classification only

        Embedding-driven objective used for discriminator update:
        - total_loss(X, Y) = l_uniform(X, Y) + gamma * l_align(X, Y)
        - X/Y are entity-id batches from positive triples (head, tail)

        negative_sampling_strategy:
        - 'multinomial': sample negatives from generator distribution
        - 'topk': choose highest-probability negatives (harder)

        emb_align_op:
        - 'add': f(e, r) = e + r
        - 'mul': f(e, r) = e * r (element-wise)

        fake_align_gamma:
        - scales repulsion from generated fake triples via (-fake align loss)
        """
        if not isinstance(train_data[0], torch.Tensor):
            train_data = [torch.LongTensor(vec) for vec in train_data]
        if not isinstance(valid_data_w_label[0], torch.Tensor):
            valid_data_w_label = [torch.LongTensor(vec) for vec in valid_data_w_label]

        # Convert to plain lists for BernCorrupterMulti so dict keys use value-based hashing
        train_data_list = [d.tolist() if isinstance(d, torch.Tensor) else d for d in train_data]

        # [ORIGINAL KBGAN]
        corrupter = BernCorrupterMulti(train_data_list, self.n_entity, self.n_relation, n_candidate)
        head, relation, tail = train_data
        n_train = len(head)
        best_perf = 0.0
        avg_reward = 0.0

        # Keep this flag for validation-time metric computation/model selection only.
        do_class_task = (class_rank_balance > 0.0)

        if emb_align_op not in ['add', 'mul']:
            raise ValueError("emb_align_op must be one of ['add', 'mul']")
        if not (0.0 <= emb_align_balance <= 1.0):
            raise ValueError("emb_align_balance must be in [0, 1]")

        # [EARLY STOPPING]
        patience_counter = 0

        for epoch in range(n_epoch):
            # [ORIGINAL KBGAN]
            epoch_emb_loss = 0.0
            epoch_reward = 0.0

            # [ORIGINAL KBGAN]
            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(n_batch, head, relation, tail, head_cand, relation_cand, tail_cand, n_sample=n_train):             
                batch_size = h.size(0)
                # --- KBGAN Generator Step ---
                gen_step, head_smpl_device, tail_smpl_device = self.generator_step(
                    hs=hs,
                    rs=rs,
                    ts=ts,
                    n_sample=n_sample,
                    temperature=temperature,
                    negative_sampling_strategy=negative_sampling_strategy,
                )

                # --- KBGAN Discriminator Step ---
                emb_loss, fake_ali_loss_reward = self.discriminator_step(
                    h=h,
                    r=r,
                    t=t,
                    head_fake=head_smpl_device,
                    tail_fake=tail_smpl_device,
                    emb_uniform_p=emb_uniform_p,
                    emb_uniform_scale=emb_uniform_scale,
                    true_align_gamma=true_align_gamma,
                    emb_align_op=emb_align_op,
                    emb_align_balance=emb_align_balance,
                    fake_align_gamma=fake_align_gamma,
                    return_fake_align=True,
                )

                rewards_raw = -fake_align_gamma * fake_ali_loss_reward
                reward_sum = float(torch.sum(rewards_raw).item())
                rewards = rewards_raw - avg_reward
                rewards_for_gen = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
                gen_step.send(rewards_for_gen)
                epoch_reward += reward_sum

                # Optimizer Step
                self.discriminator.opt_zero_grad()
                emb_loss.backward()

                # Apply entity embedding constraints (e.g. norm <= 1)
                self.discriminator.opt_step()

                # Losses are batch means, so weight by batch size then divide by n_train at epoch end.
                epoch_emb_loss += emb_loss.detach().item() * batch_size
                
            # [ORIGINAL KBGAN]       
            avg_emb_loss = epoch_emb_loss / n_train
            avg_reward = epoch_reward / n_train
            log_msg = (
                f"Train epoch {epoch + 1}/{n_epoch}, "
                f"emb_loss={avg_emb_loss:.6f}, "
                f"reward={avg_reward:.6f}"
            )
            logging.info(log_msg)

            if (epoch + 1) % epoch_per_test == 0:
                test_perf = self._run_validation_epoch(
                    epoch=epoch,
                    n_epoch=n_epoch,
                    valid_data_w_label=valid_data_w_label,
                    heads=heads,
                    tails=tails,
                    temperature=temperature,
                    do_class_task=do_class_task,
                    class_rank_balance=class_rank_balance,
                    rank_optimizing_metric=rank_optimizing_metric,
                    rank_filt=rank_filt,
                    rank_k_list=rank_k_list,
                    class_optimizing_metric=class_optimizing_metric,
                    class_use_maxgood_minbad_threshold=class_use_maxgood_minbad_threshold,
                    n_generated_valid_negative=n_generated_valid_negative,
                    class_true_percentile=class_true_percentile,
                    class_fake_percentile=class_fake_percentile,
                    class_true_fake_balance=class_true_fake_balance,
                )

                # [ORIGINAL KBGAN]
                if test_perf > best_perf:
                    best_perf = test_perf
                    self.save_kbgan()
                    print(f"Saved KBGAN at epoch {epoch + 1} with performance: {best_perf}")

                    # [EARLY STOPPING]
                    patience_counter = 0
                else:
                    # [EARLY STOPPING]
                    patience_counter += 1
                # [EARLY STOPPING]
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1} (patience={early_stop_patience})")
                    break
        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        return best_perf

    def _compute_midpoint_threshold_from_labeled_data(self, data_w_label: tuple,
                                                    n_generated_valid_negative: int=0, temperature: float=1.0,
                                                    true_percentile: float=90.0, fake_percentile: float=5.0,
                                                    true_fake_balance: float=0.5) -> float:
        """
        Compute midpoint threshold from labeled triples (and optional generator negatives)
        using percentile statistics:
        threshold = balance * percentile(true scores, true_percentile)
              + (1-balance) * percentile(fake scores, fake_percentile).
        """
        if len(data_w_label) < 4:
            return None
        
        if not (0.0 <= true_percentile <= 100.0):
            raise ValueError("true_percentile must be in [0, 100].")
        if not (0.0 <= fake_percentile <= 100.0):
            raise ValueError("fake_percentile must be in [0, 100].")

        heads_list, relations_list, tails_list, labels = data_w_label
        pos_scores = []
        neg_scores = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(
                self.discriminator.model.test_batch_size, heads_list, relations_list, tails_list, labels
            ):
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)

                # Unified threshold semantics: lower score => more likely positive.
                batch_scores = self.discriminator.score(head_var, relation_var, tail_var).detach().cpu().numpy()
                batch_labels = np.asarray(batch_label)

                pos_mask = (batch_labels == 1)
                neg_mask = (batch_labels == 0)

                if np.any(pos_mask):
                    pos_scores.extend(batch_scores[pos_mask].reshape(-1).tolist())

                if np.any(neg_mask):
                    neg_scores.extend(batch_scores[neg_mask].reshape(-1).tolist())

        if n_generated_valid_negative > 0:
            labels_array = np.asarray(labels)
            pos_mask_all = (labels_array == 1)
            if np.any(pos_mask_all):
                pos_heads = np.asarray(heads_list)[pos_mask_all].astype(np.int64)
                pos_relations = np.asarray(relations_list)[pos_mask_all].astype(np.int64)
                pos_tails = np.asarray(tails_list)[pos_mask_all].astype(np.int64)

                pos_head_tensor = torch.LongTensor(pos_heads)
                pos_relation_tensor = torch.LongTensor(pos_relations)
                pos_tail_tensor = torch.LongTensor(pos_tails)

                valid_corrupter = BernCorrupterMulti(
                    (pos_heads.tolist(), pos_relations.tolist(), pos_tails.tolist()),
                    self.n_entity,
                    self.n_relation,
                    n_generated_valid_negative,
                )
                cand_head, cand_relation, cand_tail = valid_corrupter.corrupt(
                    pos_head_tensor,
                    pos_relation_tensor,
                    pos_tail_tensor,
                    keep_truth=False,
                )

                with torch.no_grad():
                    for _, batch_relation, _, batch_hs, batch_rs, batch_ts in batch_by_size(
                        self.discriminator.model.test_batch_size,
                        pos_head_tensor,
                        pos_relation_tensor,
                        pos_tail_tensor,
                        cand_head,
                        cand_relation,
                        cand_tail,
                        n_sample=len(pos_head_tensor),
                    ):
                        gen_step = self.generator.generator_step(
                            batch_hs,
                            batch_rs,
                            batch_ts,
                            n_sample=n_generated_valid_negative,
                            temperature=temperature,
                            train=False,
                        )
                        gen_head_fake, gen_tail_fake = next(gen_step)
                        gen_head_fake = gen_head_fake.to(config.device)
                        gen_tail_fake = gen_tail_fake.to(config.device)

                        batch_relation_device = batch_relation.to(config.device)
                        if gen_head_fake.dim() == batch_relation_device.dim() + 1:
                            relation_for_fake = batch_relation_device.unsqueeze(1).expand_as(gen_head_fake)
                        else:
                            relation_for_fake = batch_relation_device

                        gen_bad_scores = self.discriminator.score(
                            gen_head_fake,
                            relation_for_fake,
                            gen_tail_fake,
                        ).detach().cpu().numpy()

                        if gen_bad_scores.size > 0:
                            neg_scores.extend(gen_bad_scores.reshape(-1).tolist())
            else:
                logging.warning(
                    "Generator validation negatives not added: validation labels contain no positive samples."
                )

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            logging.warning("Validation threshold midpoint not updated: validation labels must contain both positive and negative samples.")
            return None

        # Statistic analysis of scores
        pos_stats = _safe_stats("POS_SCORE", pos_scores)
        neg_stats = _safe_stats("NEG_SCORE", neg_scores)

        analysis_dir = os.path.join('.', 'logs', config._config.dataset, config._config.task)
        os.makedirs(analysis_dir, exist_ok=True)
        analysis_filename = config.build_timestamped_filename("valid_score_analysis", ".txt")
        analysis_path = os.path.join(analysis_dir, analysis_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write one analysis file per validation snapshot.
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("VALID SCORE ANALYSIS (LATEST)\n")
            f.write(f"valid_time: {timestamp}\n\n")

            def _write_stats(name: str, stats: dict) -> None:
                f.write(f"{name}\n")
                f.write(f"  n_sample: {stats['n_sample']}\n")
                if stats["n_sample"] == 0:
                    f.write("  min: N/A\n  max: N/A\n  mean: N/A\n  median: N/A\n  std: N/A\n")
                    f.write("  p1/p5/p10/p25/p50/p75/p90/p95/p99: N/A\n\n")
                    return

                p = stats["percentiles"]
                f.write(f"  min: {stats['min']:.6f}\n")
                f.write(f"  max: {stats['max']:.6f}\n")
                f.write(f"  mean: {stats['mean']:.6f}\n")
                f.write(f"  median: {stats['median']:.6f}\n")
                f.write(f"  std: {stats['std']:.6f}\n")
                f.write(
                    "  percentiles: "
                    f"p1={p[1]:.6f}, p5={p[5]:.6f}, p10={p[10]:.6f}, "
                    f"p25={p[25]:.6f}, p50={p[50]:.6f}, p75={p[75]:.6f}, "
                    f"p90={p[90]:.6f}, p95={p[95]:.6f}, p99={p[99]:.6f}\n\n"
                )

            _write_stats("POS_SCORE", pos_stats)
            _write_stats("NEG_SCORE", neg_stats)
        logging.info(f"Wrote latest validation score analysis to {analysis_path}")

        true_stat = float(np.percentile(np.asarray(pos_scores), true_percentile))
        fake_stat = float(np.percentile(np.asarray(neg_scores), fake_percentile))
        logging.info(
            f"Validation percentile midpoint stats:\n"
            f"\ttrue_p{true_percentile:.1f}={true_stat:.6f}\n"
            f"\tfake_p{fake_percentile:.1f}={fake_stat:.6f}"
        )
        balanced_midpoint = true_fake_balance * true_stat + (1 - true_fake_balance) * fake_stat
        return balanced_midpoint

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                    filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.evaluate_on_ranking(test_data, heads, tails,
                                                        filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_triple_classification(self, test_data_w_label: tuple,
                                          optimizing_metric: str='accuracy',
                                          use_maxgood_minbad_threshold: bool=True) -> dict:
        if not isinstance(test_data_w_label[0], torch.Tensor):
            test_data_w_label = [torch.LongTensor(vec) for vec in test_data_w_label]
            
        print("Evaluating KBGAN discriminator on Triple Classification...")
        threshold = self.optimal_threshold if use_maxgood_minbad_threshold else None
        metrics = self.discriminator.evaluate_on_classification(test_data_w_label,
                                                                optimizing_metric=optimizing_metric,
                                                                is_threshold_tunning=False, external_threshold=threshold)
        return metrics