from typing import Callable
import torch
import torch.nn.functional as F


def _combine_pair_embeddings(z_entity: torch.Tensor, z_relation: torch.Tensor, align_op: str) -> torch.Tensor:
	"""Combine entity and relation embeddings with an element-wise operation."""
	if align_op == 'add':
		return z_entity + z_relation
	if align_op == 'mul':
		return z_entity * z_relation
	raise ValueError("align_op must be one of ['add', 'mul']")

def uniform_loss(
	ids: torch.Tensor,
	emb: Callable[[torch.Tensor], torch.Tensor],
	scale: float = 2.0,
	max_sample_size: int = None,
	eps: float = 1e-12,
	) -> torch.Tensor:
	"""
	Uniformity loss over embedding spaces.

	Args:
		ids: Id batch.
		emb: Embedding function.
		scale: Distance scaling factor (default: 2.0).
		max_sample_size: Optional cap on number of ids used for uniform loss.
			If provided and len(ids) is larger, a random subset is used.
		eps: Small value to avoid log(0).

	Returns:
		Scalar tensor loss.
	"""
	if max_sample_size is not None and max_sample_size > 0 and ids.numel() > max_sample_size:
		perm = torch.randperm(ids.numel(), device=ids.device)[:max_sample_size]
		ids = ids[perm]

	z = emb(ids)
	if z.dim() != 2:
		z = z.view(z.size(0), -1)
	z = F.normalize(z, p=2, dim=1)

	# Pairwise squared Euclidean distances over all sample pairs.
	dist_sq = torch.cdist(z, z, p=2).pow(2)
	val = torch.exp(-scale * dist_sq).mean()
	return torch.log(val.clamp_min(eps))

def align_loss(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	tail_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	align_balance: float = 0.5,
	align_op: str = 'add',
	) -> torch.Tensor:
	"""
	Alignment loss on positive triples in embedding space:
		l_align((h,r), t) = E || f(emb_e(h), emb_r(r)) - emb_e(t) ||^2

	Args:
		head_ids: Head entity ids.
		relation_ids: Relation ids.
		tail_ids: Tail entity ids.
		entity_emb: Entity embedding function.
		relation_emb: Relation embedding function.
		align_balance: Balance factor in [0, 1] for head/relation before combining.
		align_op: Element-wise combine op for (head, relation), one of ['add', 'mul'].

	Returns:
		Scalar tensor loss.
	"""
	z_head = entity_emb(head_ids)
	z_rel = relation_emb(relation_ids)
	z_tail = entity_emb(tail_ids)
	if not (0.0 <= align_balance <= 1.0):
		raise ValueError("align_balance must be in [0, 1].")

	if z_head.size(0) != z_rel.size(0) or z_head.size(0) != z_tail.size(0):
		raise ValueError("head_ids, relation_ids, and tail_ids must have matching batch sizes.")
	if z_head.dim() != 2:
		z_head = z_head.view(z_head.size(0), -1)
	if z_rel.dim() != 2:
		z_rel = z_rel.view(z_rel.size(0), -1)
	if z_tail.dim() != 2:
		z_tail = z_tail.view(z_tail.size(0), -1)

	z_head = align_balance * z_head
	z_rel = (1.0 - align_balance) * z_rel
	z_pair = _combine_pair_embeddings(z_head, z_rel, align_op=align_op)
	z_pair = F.normalize(z_pair, p=2, dim=1)
	z_tail = F.normalize(z_tail, p=2, dim=1)
	return (z_pair - z_tail).pow(2).sum(dim=1).mean()