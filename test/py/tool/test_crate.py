"""Registry archives preserve build inputs and reject private inventory."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("audit_crate", ROOT / "tool/build/audit_crate.py")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class CrateTests(unittest.TestCase):
	def archive(self, extra=None):
		files = {
			"Cargo.toml": b'[package]\nname="oarust"\nversion="0.8.6"\n[lib]\nname="oa"\n',
			".oa-public-snapshot": b'public_version=0.8.6\n',
			"src/rs/lib.rs": b'', "build.rs": b'', "rustfmt.toml": b'',
			"tool/gen/fn/generate.py": b'',
			"LICENSE": b'BSL', "NOTICE.md": b'Attributions', "Cargo.lock": b'', "README.md": b'OA',
		}
		files.update(extra or {})
		directory = tempfile.TemporaryDirectory()
		self.addCleanup(directory.cleanup)
		path = Path(directory.name) / "oarust-0.8.6.crate"
		with tarfile.open(path, "w:gz") as archive:
			for name, data in files.items():
				member = tarfile.TarInfo("oarust-0.8.6/" + name)
				member.size = len(data)
				archive.addfile(member, io.BytesIO(data))
		return path

	def test_public_build_inputs_are_admitted(self):
		AUDITOR.audit(self.archive(), "0.8.6")

	def test_internal_document_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "unexpected crate path"):
			AUDITOR.audit(self.archive({"/".join(("docs", "internal", "design.md")): b"private"}), "0.8.6")
	def test_sdk_shader_build_input_is_admitted(self):
		source = "sdk/rs/slang/ml/rl/cart_pole/reset.slang"
		schema = json.dumps({"kernels": [{"source": source}]}).encode()
		AUDITOR.audit(self.archive({
			"tool/gen/fn/schema/ml/ml_training.json": schema, source: b"shader",
		}), "0.8.6")

	def test_missing_schema_shader_is_rejected(self):
		source = "sdk/rs/slang/ml/rl/cart_pole/reset.slang"
		schema = json.dumps({"kernels": [{"source": source}]}).encode()
		with self.assertRaisesRegex(ValueError, "missing crate shader input"):
			AUDITOR.audit(self.archive({"tool/gen/fn/schema/ml/ml_training.json": schema}), "0.8.6")

	def test_missing_build_script_shader_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "missing crate shader input"):
			AUDITOR.audit(self.archive({"build.rs": b'const SOURCE: &str = "src/slang/missing.slang";'}), "0.8.6")

	def test_sdk_program_source_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "unexpected crate path"):
			AUDITOR.audit(self.archive({"sdk/rs/tutorials/ml/rl/example.rs": b"program"}), "0.8.6")

	def test_declared_sdk_support_is_admitted(self):
		AUDITOR.audit(self.archive({
			"src/rs/lib.rs": b'#[path = "../../sdk/rs/mod.rs"] pub mod sdk;',
			"sdk/rs/mod.rs": b'pub mod ml;',
			"sdk/rs/ml.rs": b'pub mod rl;',
			"sdk/rs/ml/rl.rs": b'',
		}), "0.8.6")

	def test_missing_rust_path_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "missing crate Rust input: sdk/rs/mod.rs"):
			AUDITOR.audit(self.archive({
				"src/rs/lib.rs": b'#[path = "../../sdk/rs/mod.rs"] pub mod sdk;',
			}), "0.8.6")

	def test_missing_literal_rust_include_is_rejected(self):
		for macro in ("include", "include_bytes", "include_str"):
			with self.subTest(macro=macro):
				with self.assertRaisesRegex(ValueError, "missing crate Rust input"):
					AUDITOR.audit(self.archive({
						"src/rs/lib.rs": f'{macro}!("missing.rs");'.encode(),
					}), "0.8.6")

	def test_traversal_is_rejected(self):
		with self.assertRaisesRegex(ValueError, "unsafe crate path"):
			AUDITOR.audit(self.archive({"../private": b"private"}), "0.8.6")

	def test_sdk_targets_are_rejected(self):
		for target in ("bin", "example", "test", "bench"):
			with self.subTest(target=target):
				manifest = (
					f'[package]\nname="oarust"\nversion="0.8.6"\n[lib]\nname="oa"\n'
					f'[[{target}]]\nname="sdk"\npath="sdk.rs"\n'
				).encode()
				with self.assertRaisesRegex(ValueError, "SDK/integration targets"):
					AUDITOR.audit(self.archive({"Cargo.toml": manifest}), "0.8.6")

	def test_git_dependency_is_rejected(self):
		manifest = b'[package]\nname="oarust"\nversion="0.8.6"\n[lib]\nname="oa"\n[dependencies]\nbackend={git="https://example.invalid/backend"}\n'
		with self.assertRaisesRegex(ValueError, "non-registry dependency"):
			AUDITOR.audit(self.archive({"Cargo.toml": manifest}), "0.8.6")


if __name__ == "__main__":
	unittest.main()
