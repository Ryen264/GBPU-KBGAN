import os
import logging
import time
import torch
from typing import Tuple
import numpy as np

from datasets import batch_by_num, batch_by_size, BernCorrupterMulti
from component import Component, OPTIMIZER_MAP
import loss
import config

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

        # GAN fine-tuning optimizer setup with fallback to component defaults.
        self.model_config = config._config["KBGAN"]
        g_opt_name = getattr(self.model_config, 'g_optimizer', self.generator.model.optimizer_name)
        d_opt_name = getattr(self.model_config, 'd_optimizer', self.discriminator.model.optimizer_name)
        g_lr = getattr(self.model_config, 'g_learning_rate', self.generator.model.lr)
        d_lr = getattr(self.model_config, 'd_learning_rate', self.discriminator.model.lr)
        self.g_opt = OPTIMIZER_MAP.get(g_opt_name, OPTIMIZER_MAP[self.generator.model.optimizer_name])(
            self.generator.model.parameters(), lr=g_lr
        )
        self.d_opt = OPTIMIZER_MAP.get(d_opt_name, OPTIMIZER_MAP[self.discriminator.model.optimizer_name])(
            self.discriminator.model.parameters(), lr=d_lr
        )

        self.discriminator_path = self.discriminator.model_path
        self.generator_path = self.generator.model_path

        self.dataset = config._config.dataset
        self.task = config._config.task
        self.task_dir = os.path.join('.', 'models', config._config.dataset, config._config.task)
        os.makedirs(self.task_dir, exist_ok=True)

        self.model_name = 'kbgan_' + 'dis-' + self.discriminator_type + '_gen-' + self.generator_type + '.mdl'
        self.kbgan_path = os.path.join(self.task_dir, self.model_name)
        run_token = time.strftime('%y%m%d-%H%M%S') + f'-{os.getpid()}'
        self.validation_analysis_path = os.path.join(
            '.',
            'logs',
            self.dataset,
            self.task,
            f'valid_score_analysis_{run_token}.txt',
        )
        self.optimal_threshold = None
        self.best_validation_perf = None
        self.final_validation_perf = None
        self.best_validation_epoch = None
        self.training_time_seconds = None
        self.latest_validation_analysis_path = None

    def _score_summary_lines(self, title: str, scores: list) -> list:
        lines = [title]
        if len(scores) == 0:
            lines.extend([
                '  n_sample: 0',
                '  min: N/A',
                '  max: N/A',
                '  mean: N/A',
                '  median: N/A',
                '  std: N/A',
                '  percentiles: N/A',
            ])
            return lines

        values = np.asarray(scores, dtype=np.float64)
        percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        lines.extend([
            f'  n_sample: {values.size}',
            f'  min: {values.min():.6f}',
            f'  max: {values.max():.6f}',
            f'  mean: {values.mean():.6f}',
            f'  median: {np.median(values):.6f}',
            f'  std: {values.std():.6f}',
            '  percentiles: '
            f'p1={percentiles[0]:.6f}, p5={percentiles[1]:.6f}, p10={percentiles[2]:.6f}, '
            f'p25={percentiles[3]:.6f}, p50={percentiles[4]:.6f}, p75={percentiles[5]:.6f}, '
            f'p90={percentiles[6]:.6f}, p95={percentiles[7]:.6f}, p99={percentiles[8]:.6f}',
        ])
        return lines

    def _write_validation_score_analysis(self, valid_data_w_label: tuple) -> str:
        if len(valid_data_w_label) < 4:
            return None

        heads_list, relations_list, tails_list, labels = valid_data_w_label
        positive_scores = []
        negative_scores = []

        with torch.no_grad():
            for batch_head, batch_relation, batch_tail, batch_label in batch_by_size(
                self.discriminator.model.test_batch_size,
                heads_list,
                relations_list,
                tails_list,
                labels,
            ):
                head_var = torch.LongTensor(batch_head).to(config.device)
                relation_var = torch.LongTensor(batch_relation).to(config.device)
                tail_var = torch.LongTensor(batch_tail).to(config.device)
                batch_distances = self.discriminator._distance_score(head_var, relation_var, tail_var).detach().cpu().tolist()
                for distance_value, label_value in zip(batch_distances, batch_label):
                    if int(label_value) == 1:
                        positive_scores.append(float(distance_value))
                    else:
                        negative_scores.append(float(distance_value))

        log_dir = os.path.join('.', 'logs', self.dataset, self.task)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%y%m%d-%H%M%S')
        log_path = self.validation_analysis_path

        lines = [
            'VALID SCORE ANALYSIS (LATEST)',
            f'valid_time: {timestamp}',
            '',
        ]
        lines.extend(self._score_summary_lines('POS_SCORE', positive_scores))
        lines.append('')
        lines.extend(self._score_summary_lines('NEG_SCORE', negative_scores))

        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write('\n'.join(lines).rstrip() + '\n')

        logging.info('Wrote latest validation score analysis to %s', log_path)
        return log_path

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
        ) -> float:
        # [RANK TASK]
        valid_data_no_label = valid_data_w_label[:3]
        rank_metrics = self.discriminator.evaluate_on_ranking(
            valid_data_no_label,
            heads,
            tails,
            filt=rank_filt,
            k_list=rank_k_list,
        )

        # [CLASS TASK] - v4: Unaugmented Validation
        # IMPORTANT: Validation uses the raw, unmodified validation set.
        # This keeps threshold tuning aligned with the standard test distribution.
        if do_class_task:
            class_data_w_label = valid_data_w_label

            class_metrics = self.discriminator.evaluate_on_classification(
                class_data_w_label,
                optimizing_metric=class_optimizing_metric,
                is_threshold_tunning=True,
                external_threshold=None,
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
        if do_class_task:
            self.latest_validation_analysis_path = self._write_validation_score_analysis(class_data_w_label)
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
                negative_sampling_strategy: str='multinomial',
                emb_uniform_scale: float=2.0,
                entity_uniform_max_ids: int=2048,
                uniform_gamma: float=1.0,
                true_align_gamma: float=1.0,
                fake_align_gamma: float=1.0,
                safe_margin: float=1.0,
                emb_align_op: str='add',
                emb_align_balance: float=0.7,
                alpha: float=None,
                uniform_lambda: float=None,
                lambda_anchor: float=0.0,
                rank_optimizing_metric: str='mrr',
                rank_filt: bool=True,
                rank_k_list: list=[1, 3, 10],
                class_optimizing_metric: str='accuracy',
                ) -> float:
        """
        class_rank_balance is a ratio in [0, 1]:
        - 0.0 => optimize ranking only
        - 1.0 => optimize classification only

        Embedding-driven objective used for discriminator update (v4 Bounded Hybrid):
        - L_align = ||q - e_t||_2^2
        - L_fake = alpha * max(0, mu - ||q - e_t'||_2^2)
        - total_loss = L_align + L_fake + lambda * L_uni

        negative_sampling_strategy:
        - 'multinomial': sample negatives from generator distribution
        - 'topk': choose highest-probability negatives (harder)

        emb_align_op:
        - 'add': f(e, r) = e + r
        - 'mul': f(e, r) = e * r (element-wise)

        safe_margin (mu):
        - bounded margin threshold for fake negatives; prevents neighborhood shattering
        - once fake tail pushed past margin, discriminator stops pushing
        """
        config.overwrite_config_with_args([
            f"--log.prefix=KBGAN_{self.discriminator_type}_{self.generator_type}_"
        ])
        config.logger_init()

        if alpha is not None:
            fake_align_gamma = alpha
        if uniform_lambda is not None:
            uniform_gamma = uniform_lambda

        train_start = time.perf_counter()

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
        baseline_reward = 0.0
        last_validation_perf = None

        # Keep this flag for validation-time metric computation/model selection only.
        do_class_task = (class_rank_balance > 0.0)

        if emb_align_op not in ['add', 'mul']:
            raise ValueError("emb_align_op must be one of ['add', 'mul']")
        if not (0.0 <= emb_align_balance <= 1.0):
            raise ValueError("emb_align_balance must be in [0, 1]")
        if entity_uniform_max_ids is not None and entity_uniform_max_ids < 2:
            raise ValueError("entity_uniform_max_ids must be >= 2 or None")
        if uniform_gamma < 0.0:
            raise ValueError("uniform_gamma must be >= 0")
        if safe_margin <= 0.0:
            raise ValueError("safe_margin must be > 0")

        # [EARLY STOPPING]
        patience_counter = 0

        # [STEP A] Freeze pre-trained weights for anchor loss.
        # Resolve embedding weights from the wrapped model module across model types.
        def _current_anchor_weights() -> tuple[torch.Tensor, torch.Tensor]:
            model_module = self.discriminator.model.model
            if hasattr(model_module, "entity_embed") and hasattr(model_module, "relation_embed"):
                return model_module.entity_embed.weight, model_module.relation_embed.weight
            if (
                hasattr(model_module, "entity_re_embed")
                and hasattr(model_module, "entity_im_embed")
                and hasattr(model_module, "relation_re_embed")
                and hasattr(model_module, "relation_im_embed")
            ):
                entity_weight = torch.cat(
                    [model_module.entity_re_embed.weight, model_module.entity_im_embed.weight],
                    dim=-1,
                )
                relation_weight = torch.cat(
                    [model_module.relation_re_embed.weight, model_module.relation_im_embed.weight],
                    dim=-1,
                )
                return entity_weight, relation_weight
            raise AttributeError("Unsupported discriminator embedding layout for anchor loss.")

        current_entity_weight, current_relation_weight = _current_anchor_weights()
        frozen_entity_embed = current_entity_weight.detach().clone()
        frozen_relation_embed = current_relation_weight.detach().clone()

        for epoch in range(n_epoch):
            # [ORIGINAL KBGAN]
            epoch_emb_loss = 0.0
            epoch_reward = 0.0
            baseline_reward = 0.0

            # [ORIGINAL KBGAN]
            head_cand, relation_cand, tail_cand = corrupter.corrupt(head, relation, tail, keep_truth=False)
            for h, r, t, hs, rs, ts in batch_by_num(n_batch, head, relation, tail, head_cand, relation_cand, tail_cand, n_sample=n_train):             
                batch_size = h.size(0)
                # --- KBGAN Generator Step ---
                gen_step = self.generator.generator_step(
                    hs, rs, ts,
                    n_sample=n_sample,
                    temperature=temperature,
                    train=True,
                    optimizer=self.g_opt,
                    sampling_strategy=negative_sampling_strategy,
                )
                head_smpl, tail_smpl = next(gen_step)
                head_smpl_device = head_smpl.to(config.device)
                tail_smpl_device = tail_smpl.to(config.device)

                # --- KBGAN Discriminator Step ---
                h_device, r_device, t_device = h.to(config.device), r.to(config.device), t.to(config.device)
                # Batch-level uniformity over unique entities in current mini-batch.
                entity_ids_batch = torch.unique(torch.cat((h_device.reshape(-1), t_device.reshape(-1)), dim=0))
                if entity_uniform_max_ids is not None and entity_ids_batch.numel() > entity_uniform_max_ids:
                    sample_idx = torch.randperm(entity_ids_batch.numel(), device=entity_ids_batch.device)[:entity_uniform_max_ids]
                    entity_ids_batch = entity_ids_batch[sample_idx]
                ent_uni_loss = loss.uniform_loss(
                    ids=entity_ids_batch,
                    emb=self.discriminator.embed,
                    scale=emb_uniform_scale,
                )
                true_dist_sq = loss.align_distance_sq(
                    head_ids=h_device,
                    relation_ids=r_device,
                    tail_ids=t_device,
                    entity_emb=self.discriminator.embed,
                    relation_emb=self.discriminator.relation_embed,
                    attention_emb=self.discriminator.relation_attention,
                    align_balance=emb_align_balance,
                    align_op=emb_align_op,
                )

                if head_smpl_device.dim() == r_device.dim() + 1:
                    relation_for_fake = r_device.unsqueeze(1).expand_as(head_smpl_device)
                else:
                    relation_for_fake = r_device

                fake_dist_sq = loss.align_distance_sq(
                    head_ids=head_smpl_device,
                    relation_ids=relation_for_fake,
                    tail_ids=tail_smpl_device,
                    entity_emb=self.discriminator.embed,
                    relation_emb=self.discriminator.relation_embed,
                    attention_emb=self.discriminator.relation_attention,
                    align_balance=emb_align_balance,
                    align_op=emb_align_op,
                )

                # v4 Bounded Margin Discriminator Loss:
                # true pull (L_align) + bounded fake push (L_fake) + batch uniformity.
                uniform_loss_batch = ent_uni_loss
                true_pull = true_align_gamma * true_dist_sq.mean()
                # Bounded margin: max(0, mu - ||q - e_t'||_2^2)
                # Once fake is pushed past safe_margin, discriminator stops pushing (no neighborhood shattering).
                fake_loss = fake_align_gamma * torch.relu(safe_margin - fake_dist_sq).mean()

                # [STEP B] Add anchor loss to regularize embeddings toward pre-trained weights
                if lambda_anchor > 0.0:
                    current_entity_weight, current_relation_weight = _current_anchor_weights()
                    anchor_loss_e = torch.nn.functional.mse_loss(current_entity_weight, frozen_entity_embed)
                    anchor_loss_r = torch.nn.functional.mse_loss(current_relation_weight, frozen_relation_embed)
                    anchor_loss = anchor_loss_e + anchor_loss_r
                    emb_loss = true_pull + fake_loss + uniform_gamma * (uniform_loss_batch / max(1, batch_size)) + lambda_anchor * anchor_loss
                else:
                    emb_loss = true_pull + fake_loss + uniform_gamma * (uniform_loss_batch / max(1, batch_size))

                # Organic Generator Reward (v4):
                # Pure query-fooling term: -||q - e_t'||_2^2
                # Removed DENS factor-aware hardness to prevent artificial negative overlap.
                rewards_raw = -fake_dist_sq.detach()

                reward_sum = float(torch.sum(rewards_raw).item())
                rewards = rewards_raw - baseline_reward
                gen_step.send(rewards)
                epoch_reward += reward_sum
                baseline_reward = float(rewards_raw.mean().item()) if rewards_raw.numel() > 0 else 0.0

                # Optimizer Step
                self.d_opt.zero_grad()
                emb_loss.backward()

                # Apply entity embedding constraints (e.g. norm <= 1)
                self.d_opt.step()
                self.discriminator.model.constraint()

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

                )
                last_validation_perf = test_perf

                # [ORIGINAL KBGAN]
                if test_perf > best_perf:
                    best_perf = test_perf
                    self.best_validation_epoch = epoch + 1
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
            self.best_validation_perf = best_perf
            self.final_validation_perf = last_validation_perf if last_validation_perf is not None else best_perf
            self.training_time_seconds = time.perf_counter() - train_start
        print(f'Trained KBGAN successfully: {self.generator_type} generator, {self.discriminator_type} discriminator.')
        return best_perf

    def evaluate_on_link_prediction(self, heads: torch.Tensor, tails: torch.Tensor, test_data: tuple,
                                    filt: bool=True, k_list: list=[1, 3, 10]) -> dict:
        if not isinstance(test_data[0], torch.Tensor):
            test_data = [torch.LongTensor(vec) for vec in test_data]

        print("Evaluating KBGAN (discriminator) on Link Prediction...")
        metrics = self.discriminator.evaluate_on_ranking(test_data, heads, tails,
                                                        filt=filt, k_list=k_list)
        return metrics

    def evaluate_on_triple_classification(self, test_data_w_label: tuple,
                                          optimizing_metric: str='accuracy') -> dict:
        if not isinstance(test_data_w_label[0], torch.Tensor):
            test_data_w_label = [torch.LongTensor(vec) for vec in test_data_w_label]
            
        print("Evaluating KBGAN discriminator on Triple Classification...")
        # Stage-3: use relation-specific thresholds with global fallback learned in Stage-2.
        threshold = None
        metrics = self.discriminator.evaluate_on_classification(test_data_w_label,
                                                                optimizing_metric=optimizing_metric,
                                                                is_threshold_tunning=False, external_threshold=threshold)
        return metrics