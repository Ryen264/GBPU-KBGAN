from typing import Callable
import torch


def _combine_pair_embeddings(z_entity: torch.Tensor, z_relation: torch.Tensor, align_op: str) -> torch.Tensor:
	"""Combine entity and relation embeddings with an element-wise operation."""
	if align_op == 'add':
		return z_entity + z_relation
	if align_op == 'mul':
		return z_entity * z_relation
	raise ValueError("align_op must be one of ['add', 'mul']")

def uniform_loss(
	entity_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	p: float = 0.5,
	scale: float = 2.0,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""
	Uniformity loss over entity and relation embedding spaces.

	Mixed form:
		l_uniform(X, Y) = p * l_uniform(X_entities) + (1-p) * l_uniform(Y_relations)

	Args:
		entity_ids: Entity-id batch X.
		relation_ids: Relation-id batch Y.
		entity_emb: Entity embedding function.
		relation_emb: Relation embedding function.
		p: Mixing weight for entity-space uniformity in [0, 1].
		scale: Distance scaling factor (default: 2.0).
		eps: Small value to avoid log(0).

	Returns:
		Scalar tensor loss.
	"""
	if not (0.0 <= p <= 1.0):
		raise ValueError("p must be in [0, 1].")

	def _uniform_single(batch: torch.Tensor, emb_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
		z = emb_fn(batch)
		if z.dim() != 2:
			z = z.view(z.size(0), -1)

		# Pairwise squared Euclidean distances over all sample pairs.
		dist_sq = torch.cdist(z, z, p=2).pow(2)
		val = torch.exp(-scale * dist_sq).mean()
		return torch.log(val.clamp_min(eps))

	loss_x = _uniform_single(entity_ids, entity_emb)
	loss_y = _uniform_single(relation_ids, relation_emb)
	return p * loss_x + (1.0 - p) * loss_y


def align_loss(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	tail_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
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
		align_op: Element-wise combine op for (head, relation), one of ['add', 'mul'].

	Returns:
		Scalar tensor loss.
	"""
	z_head = entity_emb(head_ids)
	z_rel = relation_emb(relation_ids)
	z_tail = entity_emb(tail_ids)

	if z_head.size(0) != z_rel.size(0) or z_head.size(0) != z_tail.size(0):
		raise ValueError("head_ids, relation_ids, and tail_ids must have matching batch sizes.")

	if z_head.dim() != 2:
		z_head = z_head.view(z_head.size(0), -1)
	if z_rel.dim() != 2:
		z_rel = z_rel.view(z_rel.size(0), -1)
	if z_tail.dim() != 2:
		z_tail = z_tail.view(z_tail.size(0), -1)

	z_pair = _combine_pair_embeddings(z_head, z_rel, align_op=align_op)
	return (z_pair - z_tail).pow(2).sum(dim=1).mean()


def total_loss(
	head_ids: torch.Tensor,
	relation_ids: torch.Tensor,
	tail_ids: torch.Tensor,
	entity_emb: Callable[[torch.Tensor], torch.Tensor],
	relation_emb: Callable[[torch.Tensor], torch.Tensor],
	gamma: float,
	align_op: str = 'add',
	p: float = 0.5,
	scale: float = 2.0,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""
	Combined objective:
		total_loss(X_entities, Y_relations, tails)
		= l_uniform(X_entities, Y_relations)
		+ gamma * l_align((head, relation), tail)

	Args:
		head_ids: Head entity ids (X).
		relation_ids: Relation ids (Y).
		tail_ids: Tail entity ids.
		entity_emb: Entity embedding function.
		relation_emb: Relation embedding function.
		gamma: Weight for alignment term.
		align_op: Element-wise combine op for (head, relation), one of ['add', 'mul'].
		p: Mixing weight used in l_uniform(entities, relations).
		scale: Distance scaling for uniform loss.
		eps: Numerical stability epsilon for uniform loss.

	Returns:
		Scalar tensor total loss.
	"""
	u = uniform_loss(
		entity_ids=head_ids,
		relation_ids=relation_ids,
		entity_emb=entity_emb,
		relation_emb=relation_emb,
		p=p,
		scale=scale,
		eps=eps,
	)
	a = align_loss(
		head_ids=head_ids,
		relation_ids=relation_ids,
		tail_ids=tail_ids,
		entity_emb=entity_emb,
		relation_emb=relation_emb,
		align_op=align_op,
	)
	return u + gamma * a