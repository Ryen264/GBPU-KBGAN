# KBGAN Fixes - Quick Reference

## The Two Critical Bugs

### Bug #1: Generator Gradient Flow Broken 🔴

**Location**: `kbgan.py` line 163, in `generator_step()` method

**BEFORE (Broken)**:
```python
reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.to(config.device), sample_idx.data])
```
❌ **Problem**: 
- `sample_idx.data` is CPU tensor
- `log_probs` is GPU tensor  
- Indexing GPU tensor with CPU indices → **Device Mismatch**
- Generator gradients never flow
- **Generator NEVER learns**

**AFTER (Fixed)**:
```python
row_idx_device = row_idx.to(config.device)
sample_idx_device = sample_idx.to(config.device)
rewards_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, device=config.device, dtype=torch.float32)
if rewards_tensor.device != config.device:
    rewards_tensor = rewards_tensor.to(config.device)
reinforce_loss = -torch.sum(rewards_tensor * log_probs[row_idx_device, sample_idx_device])
```
✅ **Result**:
- All tensors on correct device
- Proper indexing
- **Generator gradients flow correctly**

---

### Bug #2: Hard Negative Mining Disabled 🔴

**Location**: `main.py` line 40, KBGAN configuration parsing

**BEFORE (Broken)**:
```python
n_candidate = _config['KBGAN'].get('n_candidate', n_sample)
# If n_sample = 20, then n_candidate defaults to 20
# topk(k=20, dim=-1) on 20 candidates = return ALL 20 (sorted)
# NO HARD MINING HAPPENS!
```
❌ **Problem**: 
- topk(20) on 20 candidates returns ALL candidates
- No selection of hardest negatives
- Same weak training signal every epoch
- **Discriminator can't improve**

**AFTER (Fixed)**:
```python
negative_sampling_strategy = _config['KBGAN'].get('negative_sampling_strategy', 'topk')
if negative_sampling_strategy == 'topk':
    # Hard mining: default pool size = 100 to select top n_sample=20
    n_candidate = _config['KBGAN'].get('n_candidate', 100)
else:
    # Multinomial: pool size equals sample size (no hard mining)
    n_candidate = _config['KBGAN'].get('n_candidate', n_sample)
```
✅ **Result**:
- topk(20) on 100 candidates = select TOP 20 hardest
- **Real hard negative mining active**
- Stronger training signal

---

## Impact on Training

### What Was Happening (Before Fixes)

```
Epoch 1-500: 
- Rank_Loss = 0.324 ↔ 0.328 (STUCK)
  └─ Why? No hard negatives → margin already ~satisfied
  └─ Generator not learning → always random fakes
  
- Class_Loss = 5.354 ↔ 5.352 (STUCK)
  └─ Why? Weak gradients from easy negatives
  └─ Generator not updating → no progress signal
  
- Validation Performance = NO IMPROVEMENT
```

### What Should Happen (After Fixes)

```
Epoch 1-50:
- Rank_Loss = 0.322 → 0.280 → 0.240 → ... ⬇️ DECREASING
  └─ Why? Hard negatives force margin violations
  └─ Generator learning to create tougher fakes
  
- Class_Loss = 5.354 → 4.900 → 4.500 → ... ⬇️ DECREASING  
  └─ Why? Strong gradients from hard negatives
  └─ Discriminator learning clearer discrimination
  
- Validation Performance = IMPROVING ✨
```

---

## How to Verify Fixes Are Working

### Check #1: Generator Is Updating
Add this debug code to `train_kbgan()` after `gen_step.send()`:
```python
# Check if generator parameters changed
gen_param_sum_before = sum(p.sum().item() for p in self.generator.model.parameters())
# ... training step ...
gen_param_sum_after = sum(p.sum().item() for p in self.generator.model.parameters())
if gen_param_sum_before != gen_param_sum_after:
    logging.debug("✅ Generator parameters updated!")
else:
    logging.debug("❌ Generator parameters UNCHANGED!")
```

### Check #2: Hard Mining Active
Look at training logs for:
```
n_candidate=100, n_sample=20, negative_sampling_strategy=topk
       ↑
   Should say 100, not 20!
```

### Check #3: Losses Decreasing
Expected in first 50 epochs:
```
Epoch 1, Rank_Loss=0.3220, Class_Loss=5.3541
Epoch 10, Rank_Loss=0.2950, Class_Loss=5.1200  ← decreasing
Epoch 20, Rank_Loss=0.2480, Class_Loss=4.7800  ← still decreasing
Epoch 50, Rank_Loss=0.1200, Class_Loss=3.2100  ← clear trend
```

If NOT decreasing → there's still an issue

---

## Configuration Example

### wn18rr config with fixes activated

```yaml
KBGAN:
  class_rank_balance: 0.2
  early_stop_patience: -1
  temperature: 1.0
  n_sample: 20
  # n_candidate will AUTO-DEFAULT to 100 for topk strategy
  negative_sampling_strategy: 'topk'  ← Triggers hard mining
  n_epoch: 5000
  n_batch: 100
  epoch_per_test: 100
  class_rank_balance_start: 0.1
  class_rank_balance_warmup_epochs: 15
  loss_join_method: 'adaptive_weight'
```

**Result**: 
- ✅ Hard mining enabled (100→20)
- ✅ Generator updates working (device fix)
- ✅ Losses should decrease meaningfully

---

## Common Questions

**Q: Why were losses stuck?**  
A: Two reasons: (1) Generator not training due to device bug, (2) Negatives not hard due to n_candidate=n_sample

**Q: Will losses go to zero?**  
A: Probably not to exactly zero, but should improve significantly. Stabilizes when margin satisfiedmost of the time.

**Q: How can I make mining even harder?**  
A: Increase n_candidate: `n_candidate: 200` (select 20 from 200)

**Q: What if I use multinomial sampling?**  
A: Hard mining not applicable. Set n_candidate = n_sample to use uniform sampling.

**Q: Do I need to retrain components?**  
A: No, these fixes only affect KBGAN training. Pre-trained components still usable.

