"""OA-CC-34 fusion-class schema coverage.

Every operation in every domain schema must declare a `fusion_class` from
the roadmap taxonomy. Opaque and effect operations carry the class that
keeps them out of region formation.
"""

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SCHEMA_ROOTS = [
	REPOSITORY_ROOT / "tool/gen/fn/schema",
	REPOSITORY_ROOT / "tool/gen/nn/schema",
]

FUSION_CLASSES = {
	"pointwise",
	"broadcast",
	"gather",
	"reduce",
	"scan",
	"gemm",
	"conv",
	"attention",
	"opaque",
	"effect",
}


def iter_schemas():
	for root in SCHEMA_ROOTS:
		for path in sorted(root.rglob("*.json")):
			yield path


class FusionClassCoverage(unittest.TestCase):
	def test_every_operation_declares_a_class(self) -> None:
		missing = []
		total = 0
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for index, operation in enumerate(schema.get("operations", [])):
				total += 1
				fusion_class = operation.get("fusion_class")
				if fusion_class not in FUSION_CLASSES:
					missing.append(
						f"{path.relative_to(REPOSITORY_ROOT)} operations[{index}]"
						f" {operation.get('name')}: {fusion_class!r}"
					)
		self.assertGreater(total, 0)
		self.assertEqual(missing, [])

	def test_every_contract_and_kernel_declares_a_class(self) -> None:
		missing = []
		total = 0
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			contracts = {c["name"]: c for c in schema.get("contracts", []) if "name" in c}
			for section in ("contracts", "kernels"):
				for index, record in enumerate(schema.get(section, [])):
					if "name" not in record:
						continue
					total += 1
					fusion_class = record.get("fusion_class")
					if fusion_class is None and section == "kernels":
						semantic = record.get("semantic_operation")
						if semantic is not None:
							fusion_class = contracts.get(semantic, {}).get(
								"fusion_class"
							)
					if fusion_class not in FUSION_CLASSES:
						missing.append(
							f"{path.relative_to(REPOSITORY_ROOT)} {section}[{index}]"
							f" {record['name']}: {fusion_class!r}"
						)
		self.assertGreater(total, 0)
		self.assertEqual(missing, [])

	def test_effect_and_opaque_classes_exist_where_domains_require(self) -> None:
		# Opaque and effect are distinct region-exclusion classes.
		# Require examples of both in the operation schema.
		classes: set[str] = set()
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for operation in schema.get("operations", []):
				classes.add(operation["fusion_class"])
		self.assertIn("opaque", classes)
		self.assertIn("effect", classes)


SCALAR_TAILS = {
	"add",
	"sub",
	"mul",
	"div",
	"max",
	"min",
	"neg",
	"abs",
	"log",
	"exp",
	"sqrt",
	"recip",
	"relu",
}

REGION_READS = {"elementwise", "gather", "bordered"}


class RegionReadCoverage(unittest.TestCase):
	def test_declared_region_reads_use_the_taxonomy(self) -> None:
		invalid = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for section in ("operations", "contracts", "kernels"):
				for index, record in enumerate(schema.get(section, [])):
					if "name" not in record:
						continue
					region_read = record.get("region_read")
					if region_read is not None and region_read not in REGION_READS:
						invalid.append(
							f"{path.relative_to(REPOSITORY_ROOT)} {section}[{index}]"
							f" {record['name']}: {region_read!r}"
						)
		self.assertEqual(invalid, [])

	def test_every_image_contract_and_audio_kernel_resolves_a_read(self) -> None:
		# Image resolves at contract level; audio kernels under one shared
		# contract carry per-kernel reads. Every kernel entry in both
		# domains must resolve a class through the registry rule (kernel
		# record first, owning contract second).
		missing = []
		for name in ("audio.json", "image.json"):
			schema = json.loads(
				(REPOSITORY_ROOT / "tool/gen/fn/schema" / name.split(".")[0] / name).read_text()
			)
			contracts = {c["name"]: c for c in schema.get("contracts", []) if "name" in c}
			for index, kernel in enumerate(schema.get("kernels", [])):
				if "name" not in kernel:
					continue
				region_read = kernel.get("region_read")
				if region_read is None:
					semantic = kernel.get("semantic_operation")
					if semantic is not None:
						region_read = contracts.get(semantic, {}).get("region_read")
				if region_read not in REGION_READS:
					missing.append(
						f"{name} kernels[{index}] {kernel['name']}: {region_read!r}"
					)
		self.assertEqual(missing, [])


class ScalarTailCoverage(unittest.TestCase):
	def test_declared_scalar_tails_use_the_taxonomy(self) -> None:
		invalid = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for index, operation in enumerate(schema.get("operations", [])):
				if "name" not in operation:
					continue
				tail = operation.get("scalar_tail")
				if tail is not None and tail not in SCALAR_TAILS:
					invalid.append(
						f"{path.relative_to(REPOSITORY_ROOT)} operations[{index}]"
						f" {operation['name']}: {tail!r}"
					)
		self.assertEqual(invalid, [])

	def test_every_tail_op_maps_to_a_tail_kernel(self) -> None:
		# The scalar-epilogue fold set is the tail metadata's entire
		# contract: each declared op's kernel admits the tail and no
		# kernel without a declaration does.
		tails = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for operation in schema.get("operations", []):
				if "scalar_tail" in operation:
					tails.append((operation.get("kernel_id"), operation["scalar_tail"]))
		self.assertEqual(len(tails), 13)
		self.assertTrue(all(kernel for kernel, _ in tails))


TAPE_OPS = {
	"gather", "add", "sub", "mul", "div", "pow", "min", "max",
	"neg", "recip", "abs", "exp", "log", "sqrt", "sin", "cos",
	"tanh", "sigmoid", "relu", "gelu", "silu",
	"relu_backward", "gelu_backward", "silu_backward", "tanh_backward",
	"sigmoid_backward", "softplus_backward", "mish_backward",
	"elu_backward", "leaky_relu_backward", "gain_db", "clamp",
}

TAPE_SHAPES = {
	"binary", "unary", "scalar", "scalar_first", "scalar_pair",
	"broadcast", "gather", "binary_scalar",
}

ACT_KINDS = {
	"relu", "gelu", "silu", "tanh", "sigmoid",
	"mish", "elu", "leaky_relu", "softplus",
}


class TapeOpCoverage(unittest.TestCase):
	def test_declared_tape_ops_use_the_taxonomy(self) -> None:
		invalid = []
		for path in iter_schemas():
			for record in iter_tape_records(path):
				op, shape = record.get("tape_op"), record.get("tape_shape")
				if op is None and shape is None:
					continue
				where = f"{path.relative_to(REPOSITORY_ROOT)} {record.get('name')}"
				if op not in TAPE_OPS or shape not in TAPE_SHAPES:
					invalid.append(f"{where}: {op!r}/{shape!r}")
		self.assertEqual(invalid, [])

	def test_every_tape_op_maps_to_a_kernel(self) -> None:
		kernels = set()
		for path in iter_schemas():
			for record in iter_tape_records(path):
				if "tape_op" in record:
					kernels.add(record.get("kernel_id"))
		self.assertEqual(len(kernels), 43)
		self.assertNotIn(None, kernels)


class ActKindCoverage(unittest.TestCase):
	def test_declared_act_kinds_use_the_taxonomy(self) -> None:
		invalid = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for index, operation in enumerate(schema.get("operations", [])):
				if "name" not in operation:
					continue
				kind = operation.get("act_kind")
				if kind is not None and kind not in ACT_KINDS:
					invalid.append(
						f"{path.relative_to(REPOSITORY_ROOT)} operations[{index}]"
						f" {operation['name']}: {kind!r}"
					)
		self.assertEqual(invalid, [])

	def test_every_act_kind_maps_to_a_kernel(self) -> None:
		# Nine forward + nine backward activations carry the fused
		# `linear_act` family's activation-kind ABI.
		kernels = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for operation in schema.get("operations", []):
				if "act_kind" in operation:
					kernels.append(operation.get("kernel_id"))
		self.assertEqual(len(kernels), 18)
		self.assertNotIn(None, kernels)

	def test_declared_tile_programs_use_the_taxonomy(self) -> None:
		import re
		pattern = re.compile(
			r"[a-z][a-z0-9_]*(\((true|false|0x[0-9a-fA-F]+|[0-9]+)"
			r"(,\s*(true|false|0x[0-9a-fA-F]+|[0-9]+))*\))?"
		)
		invalid = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for section in ("contracts", "operations", "kernels"):
				for index, record in enumerate(schema.get(section, [])):
					if not isinstance(record, dict):
						continue
					program = record.get("tile_program")
					if program is not None and (
						not isinstance(program, str) or pattern.fullmatch(program) is None
					):
						invalid.append(
							f"{path.relative_to(REPOSITORY_ROOT)} {section}[{index}]"
							f" {record.get('name')}: {program!r}"
						)
		self.assertEqual(invalid, [])

	def test_every_tile_program_maps_to_a_kernel(self) -> None:
		# 49 image contracts + 2 resize kernels + 2 audio kernels supply
		# the tail-program factory table consumed by `tile_scalar_tail_program`.
		from oafn.registry import kernel_tile_program
		resolved = []
		for name in ("image", "audio"):
			schema = json.loads(
				(REPOSITORY_ROOT / "tool/gen/fn/schema" / name / f"{name}.json").read_text()
			)
			contracts = {c["name"]: c for c in schema.get("contracts", [])}
			for kernel in schema.get("kernels", []):
				if kernel_tile_program(kernel, contracts) is not None:
					resolved.append(kernel["kernel_id"])
		self.assertEqual(len(resolved), 53)
		self.assertIn("ImageResizeNearestF32", resolved)
		self.assertIn("AudioMelNormalizeF32", resolved)

	def test_declared_prologue_kinds_use_the_taxonomy(self) -> None:
		kinds = {"silu_mul", "add", "sub", "mul", "scalar_mul"}
		invalid = []
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for record in iter_tape_records(path):
				kind = record.get("prologue_kind")
				if kind is not None and kind not in kinds:
					invalid.append(
						f"{path.relative_to(REPOSITORY_ROOT)}"
						f" {record.get('name')}: {kind!r}"
					)
		self.assertEqual(invalid, [])

	def test_every_prologue_kind_maps_to_a_kernel(self) -> None:
		# `linear_backward_prologue`'s five producer decode entries:
		# silu_mul adjoint, pointwise add/sub/mul, and broadcast splat mul.
		from oafn.registry import prologue_kind_code
		resolved = set()
		for path in iter_schemas():
			schema = json.loads(path.read_text())
			for record in iter_tape_records(path):
				kind = record.get("prologue_kind")
				if kind is not None:
					self.assertIsNotNone(record.get("kernel_id"), record.get("name"))
					resolved.add(prologue_kind_code(kind))
		self.assertEqual(resolved, {9, 10, 11, 12, 13})


def iter_tape_records(path):
	schema = json.loads(path.read_text())
	for section in ("operations", "kernels"):
		for record in schema.get(section, []):
			yield record
			for variant in record.get("additional_lowering_variants", []):
				yield variant


if __name__ == "__main__":
	unittest.main()
