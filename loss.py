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

def _normalize_rows(z: torch.Tensor) -> torch.Tensor:
	return F.normalize(z, p=2, dim=-1)

def uniform_loss(
	ids: torch.Tensor,
	emb: Callable[[torch.Tensor], torch.Tensor],
	scale: float = 2.0,
	eps: float = 1e-12,
	) -> torch.Tensor:
	"""
	Uniformity loss over embedding spaces.

	Args:
		ids: Id batch.
		emb: Embedding function.
		scale: Distance scaling factor (default: 2.0).
		eps: Small value to avoid log(0).

	Returns:
		Scalar tensor loss.
	"""
	z = emb(ids)
	if z.dim() != 2:
		z = z.view(z.size(0), -1)
	z = F.normalize(z, p=2, dim=1)
	if z.size(0) < 2:
		# No pairs available; return neutral uniformity contribution.
		return z.new_zeros(())

	# Use pdist to avoid materializing an NxN matrix.
	dist_sq = torch.pdist(z, p=2).pow(2)
	val = torch.exp(-scale * dist_sq).mean()
	return torch.log(val.clamp_min(eps))

def compose_query(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	attention_emb: Callable[[torch.Tensor], torch.Tensor] = None,
	align_balance: float = 0.5,
	align_op: str = 'add',
	) -> torch.Tensor:
	"""Compose and normalize query embedding q from (head, relation)."""
	if attention_emb is None and not (0.0 <= align_balance <= 1.0):
		raise ValueError("align_balance must be in [0, 1].")

	z_head = entity_emb(head_ids)
	z_rel = relation_emb(relation_ids)
	if z_head.shape != z_rel.shape:
		raise ValueError("head_ids and relation_ids must map to embeddings of identical shape.")

	z_head = _normalize_rows(z_head)
	z_rel = _normalize_rows(z_rel)
	if attention_emb is not None:
		z_attention = torch.sigmoid(attention_emb(relation_ids))
		if z_attention.shape != z_head.shape:
			raise ValueError("relation attention weights must have the same shape as entity embeddings.")
		z_head = _normalize_rows(z_head * z_attention)
		z_query = _combine_pair_embeddings(z_head, z_rel, align_op='mul')
		return _normalize_rows(z_query)

	z_head = align_balance * z_head
	z_rel = (1.0 - align_balance) * z_rel
	z_query = _combine_pair_embeddings(z_head, z_rel, align_op=align_op)
	return _normalize_rows(z_query)

def align_distance_sq(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	tail_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	attention_emb: Callable[[torch.Tensor], torch.Tensor] = None,
	align_balance: float = 0.5,
	align_op: str = 'add',
	) -> torch.Tensor:
	"""Per-sample squared distance ||q - t||^2 used by DirectAU-style losses."""
	z_query = compose_query(
		head_ids=head_ids,
		relation_ids=relation_ids,
		entity_emb=entity_emb,
		relation_emb=relation_emb,
		attention_emb=attention_emb,
		align_balance=align_balance,
		align_op=align_op,
	)
	z_tail = _normalize_rows(entity_emb(tail_ids))
	if z_query.shape != z_tail.shape:
		raise ValueError("(head, relation) query embeddings and tail embeddings must have identical shape.")
	return (z_query - z_tail).pow(2).sum(dim=-1)

def align_loss(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	tail_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	attention_emb: Callable[[torch.Tensor], torch.Tensor] = None,
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
	return align_distance_sq(
		head_ids=head_ids,
		relation_ids=relation_ids,
		tail_ids=tail_ids,
		entity_emb=entity_emb,
		relation_emb=relation_emb,
		attention_emb=attention_emb,
		align_balance=align_balance,
		align_op=align_op,
	).mean()