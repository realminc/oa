"""Conservative physical effects for compute-region admission.

Logical OpEffect reads/writes, replay safety and physical-write ownership
remain independent contracts. An audited empty fusion_effects array permits
resource proofs; missing metadata never establishes purity.
"""

from __future__ import annotations

from typing import Any

from .util import require


FUSION_EFFECTS = {
	"unknown": "UNKNOWN",
	"random": "RANDOM",
	"state_mutation": "STATE_MUTATION",
	"atomic": "ATOMIC",
	"host": "HOST",
	"execution_boundary": "EXECUTION_BOUNDARY",
}


def kernel_fusion_effects(
	record: dict[str, Any], *, inherited: dict[str, Any] | None = None
) -> list[str]:
	"""Normalize an audit plus non-overridable existing effect evidence.

	Variants require their own audit. Parent mutation/replay facts still apply;
	parent purity does not. Workgroup-local barriers are not graph boundaries.
	"""
	label = f"kernel {record.get('kernel_id', record.get('name', '<unnamed>'))}.fusion_effects"
	declared = record.get("fusion_effects", ["unknown"])
	require(isinstance(declared, list), f"{label} must be an array")
	require(
		all(isinstance(effect, str) and effect in FUSION_EFFECTS for effect in declared),
		f"{label} contains an unsupported effect",
	)
	require(len(declared) == len(set(declared)), f"{label} contains duplicate effects")
	# A variant cannot erase known parent effects, but an independently audited
	# variant may resolve a parent's lack of classification.
	inferred = set(kernel_fusion_effects(inherited)) - {"unknown"} if inherited else set()
	for source in (inherited or {}, record):
		role = source.get("training_replay_role", "safe")
		if role in {"frozen_rng", "replay_rng"}:
			inferred.add("random")
		if role in {
			"host_stepped_optimizer", "optimizer_state_advance", "optimizer_state_update",
			"rng_state_advance",
		} or source.get("contract", {}).get("mutated_inputs"):
			inferred.add("state_mutation")
		physical = source.get("physical_write") or {}
		writes = [*physical.get("writes", [])]
		if physical.get("uniform_stores") is not None:
			writes.append(physical["uniform_stores"])
		if any(
			write.get("collision") == "atomic_u32"
			or write.get("partition") == "shared_atomic_contributors"
			for write in writes
		):
			inferred.add("atomic")
	require(
		declared != [] or not inferred,
		f"{label} declares purity but existing metadata requires {', '.join(sorted(inferred))}",
	)
	combined = set(declared) | inferred
	return [effect for effect in FUSION_EFFECTS if effect in combined]


def rust_fusion_effects(effects: list[str]) -> str:
	if not effects:
		return "KernelEffects::empty()"
	expression = f"KernelEffects::{FUSION_EFFECTS[effects[0]]}"
	for effect in effects[1:]:
		expression += f".union(KernelEffects::{FUSION_EFFECTS[effect]})"
	return expression
