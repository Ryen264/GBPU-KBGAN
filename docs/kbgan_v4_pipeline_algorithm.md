# KBGAN v4 Full Pipeline Algorithm

This document is the canonical reference for the current KBGAN v4 pipeline.
It matches the implementation in `kbgan.py` and `main.py`.

## Inputs

- Training set `T_train = {(h, r, t)}`
- Validation set `T_val` with raw, unmodified labels
- Test set `T_test`
- Pretrained generator `G` with parameters `theta_G`
- Discriminator `D` with parameters `theta_D`
- Relation attention weights `w in R^{|R| x d}`
- Hyperparameters:
  - `alpha`: fake alignment weight
  - `gamma`: uniformity weight
  - `mu`: safe margin
  - `temperature`
  - `n_sample`
  - `n_candidate`
  - `class_rank_balance`

## Stage 1: Adversarial Training

1. Initialize the generator baseline `b = 0`.
2. Initialize relation attention weights `w`.
3. Repeat for each training epoch:
4. Sample a mini-batch `T_batch` from `T_train`.
5. Build the batch entity set `E_batch` from the unique heads and tails in the batch.
6. Compute batch uniformity regularization on `E_batch`.
7. For each triple `(h, r, t)` in `T_batch`:
8. Extract normalized base embeddings `e_h`, `e_r`, and `e_t` from the discriminator.
9. Apply relation attention to the head embedding and re-normalize.
10. Compose the query `q` from the attention-masked head and relation embedding, then re-normalize.
11. Sample one fake tail `t_s'` from the generator.
12. Extract the discriminator embedding for `t_s'`.
13. Compute the true alignment loss `L_align = ||q - e_t||_2^2`.
14. Compute the bounded fake loss `L_fake = alpha * relu(mu - ||q - e_t'||_2^2)`.
15. Add batch uniformity: `L_D = L_align + L_fake + (gamma / |T_batch|) * L_uni`.
16. Backpropagate `L_D` through the discriminator and relation attention weights.
17. Compute the generator reward `r = -||q - e_t'||_2^2`.
18. Accumulate the policy-gradient baseline with the batch mean reward.
19. Backpropagate the REINFORCE loss through the generator using `(r - b)`.
20. Update discriminator, generator, and attention parameters.
21. Update the baseline to the mean reward for the current batch.
22. Continue until the configured epoch budget is reached or early stopping triggers.

### Training notes

- The discriminator only pushes fake tails while they are inside the safe margin.
- The generator is rewarded purely for fooling the query, without structural similarity penalties.
- Validation during training always uses the raw validation set, not an augmented version.

## Stage 2: Validation and Thresholding

1. Use `T_val` exactly as stored on disk.
2. For each relation that appears in validation, compute distances for all labeled validation triples.
3. Search relation-specific thresholds `delta_r` that maximize the selected metric, typically accuracy.
4. Also search a global fallback threshold `delta_global` for unseen relations.
5. Store the best thresholds when the validation score improves.

## Stage 3: Inference

### Link Prediction

1. Build the query `q` for `(h, r, ?)`.
2. Score every entity with `d_e = ||q - e||_2^2`.
3. Rank entities in ascending order of distance.
4. Apply filtered ranking when requested.

### Triple Classification

1. Build the query `q` for `(h, r, t)`.
2. Compute `d_test = ||q - e_t||_2^2`.
3. Predict positive when `d_test <= delta_r`, or `delta_global` when the relation is unseen.

## Implementation Mapping

- Training loop: `kbgan.py`, `train_kbgan()`
- Validation loop: `kbgan.py`, `_run_validation_epoch()`
- Threshold tuning: `component.py`, `evaluate_on_classification()`
- Run summary: `main.py`, `_write_summary_report()`