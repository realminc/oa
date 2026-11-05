"""Output inventory assembly from loaded schemas."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .audio import generate_audio_api
from .bigint import bigint_outputs
from .crypto import generate_cryptography_api
from .hash import (
	cryptography_hash_binding_outputs,
	generate_fixed_hash_shader,
	generate_merkle_proof_shader,
	generate_poseidon2_constants,
	generate_poseidon2_shader,
)
from .image import IMAGE_CATEGORIES, generate_image_api
from .matrix import (
	generate_api,
	generate_blas_api,
	generate_blas_candidate_shader,
	generate_blas_shader,
	generate_blas_test,
	generate_reduce_api,
	generate_reduce_test,
	generate_shader,
	generate_test,
	matrix_autograd_outputs,
)
from .merkle import merkle_outputs
from .misc import SESSION_TYPE_PATHS
from .ml import (
	generate_ml_activation_api,
	generate_ml_autograd_attachments,
	generate_ml_autograd_records,
	generate_ml_autograd_routing,
	generate_ml_matrix_family,
)
from .model import operation_variants
from .ntt import ntt_outputs
from .paths import FN_ROOT
from .pqc import (
	generate_mldsa_keygen_shader,
	generate_mldsa_prehash_digest_shader,
	generate_mldsa_prehash_layout_shader,
	generate_mldsa_prehash_shader,
	generate_mldsa_sign_shader,
	generate_mldsa_sign_workspace_layout_shader,
	generate_mldsa_verify_shader,
	generate_mlkem_kem_shader,
	generate_mlkem_keygen_shader,
	generate_mlkem_shared_derive_shader,
	pqc_device_outputs,
)
from .python import (
	generate_native_stub,
	generate_python_domain,
	merge_python_fragments,
	python_struct_results,
)
from .registry import (
	generate_dnn_roles,
	generate_kernel_manifest,
	generate_operation_registry,
)
from .schema import SCHEMAS, Schemas, ensure_schemas
from .signature import (
	domain_signatures,
	generate_domain_methods,
	generate_signature_parity,
)
from .types import generate_types_file
from .util import banner, format_rust
from .vision import generate_vision_api


def expected_outputs(root: Path, schemas: Schemas) -> dict[Path, str]:
	schemas = ensure_schemas(schemas)
	elementwise, elementwise_hash = schemas["elemwise"]
	blas, blas_hash = schemas["blas"]
	reduce, reduce_hash = schemas["reduce"]
	rng, rng_hash = schemas["rng"]
	index, index_hash = schemas["index"]
	view, view_hash = schemas["view"]
	ml, ml_hash = schemas["ml"]
	activation, activation_hash = schemas["ml_activation"]
	audio, audio_hash = schemas["audio"]
	cryptography, cryptography_hash = schemas["hash"]
	merkle, merkle_hash = schemas["merkle"]
	ntt, ntt_hash = schemas["ntt"]
	cryptography_pqc, cryptography_pqc_hash = schemas["pqc"]
	bigint, bigint_hash = schemas["bigint"]
	image, image_hash = schemas["image"]
	vision, vision_hash = schemas["vision"]
	sessions, sessions_hash = schemas["sessions"]
	codecs, codecs_hash = schemas["codecs"]
	ml_parity, ml_parity_hash = schemas["ml_parity"]
	exports, exports_hash = schemas["exports"]
	outputs = {
		root / "src/rs/core/operation/operation.gen.rs": format_rust(
			generate_operation_registry(schemas),
			root,
		),
		root / "src/rs/matrix/elemwise.gen.rs": format_rust(generate_api(elementwise, elementwise_hash), root),
		root / "src/rs/matrix/blas.gen.rs": format_rust(generate_blas_api(blas, blas_hash), root),
		root / "src/rs/matrix/reduce.gen.rs": format_rust(
			generate_reduce_api(reduce, reduce_hash), root
		),
		root / "src/rs/matrix/matrix.gen.rs": format_rust(
			generate_domain_methods(
				"matrix",
				domain_signatures("matrix", [elementwise, blas, reduce, rng, index, view]),
				hashlib.sha256((elementwise_hash + blas_hash + reduce_hash + rng_hash + index_hash + view_hash).encode()).hexdigest(),
				"matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,matrix_rng.json,matrix_index.json,matrix_view.json",
			),
			root,
		),
		root / "src/rs/runtime/dna/kernel/kernel.gen.rs": format_rust(
			generate_kernel_manifest(schemas), root
		),
		root / "src/rs/runtime/dna/provider/dnn/generated.rs": format_rust(
			generate_dnn_roles(schemas), root
		),
		root / "test/rs/matrix/test_elemwise.gen.rs": format_rust(generate_test(elementwise, elementwise_hash), root),
		root / "test/rs/matrix/test_blas.gen.rs": format_rust(generate_blas_test(blas, blas_hash), root),
		root / "test/rs/matrix/test_reduce.gen.rs": format_rust(
			generate_reduce_test(reduce, reduce_hash), root
		),
		root / "src/rs/ml/matrix/activation.gen.rs": format_rust(
			generate_ml_activation_api(activation, activation_hash), root
		),
	}
	for name, (types_schema, types_hash) in sorted(schemas.items()):
		if name.endswith("_types"):
			domain = types_schema["domain"]
			outputs[root / f"src/rs/{domain}/type.gen.rs"] = format_rust(
				generate_types_file(types_schema, types_hash, SCHEMAS[name].path.name), root
			)
	outputs.update(matrix_autograd_outputs(root, [(elementwise, elementwise_hash), (blas, blas_hash), (reduce, reduce_hash), (rng, rng_hash), (index, index_hash), (view, view_hash)]))
	outputs[root / "src/rs/ml/matrix/swiglu.gen.rs"] = format_rust(
		generate_ml_activation_api(activation, activation_hash, "swiglu"), root
	)
	for family in sorted({op["rust_family"] for op in ml["operations"] if "rust_family" in op}):
		outputs[root / f"src/rs/ml/matrix/{family}.gen.rs"] = format_rust(
			generate_ml_matrix_family(ml, ml_hash, family), root
		)
	outputs[root / "src/rs/crypto/hash/batch.gen.rs"] = format_rust(generate_cryptography_api(cryptography, cryptography_hash, "hash"), root)
	outputs.update(cryptography_hash_binding_outputs(root, cryptography, cryptography_hash))
	outputs.update(bigint_outputs(root, bigint, bigint_hash))
	outputs[root / "src/rs/crypto/bigint/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			"crate::crypto::bigint",
			domain_signatures("crypto::bigint", [bigint]),
			bigint_hash,
			"crypto_bigint.json",
		),
		root,
	)
	outputs.update(ntt_outputs(root, ntt, ntt_hash))
	outputs[root / "src/rs/crypto/ntt/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			"crate::crypto::ntt",
			domain_signatures("crypto::ntt", [ntt]),
			ntt_hash,
			"crypto_ntt.json",
		),
		root,
	)
	outputs.update(merkle_outputs(root, merkle, merkle_hash, cryptography))
	outputs[root / "src/rs/crypto/merkle/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			"crate::crypto::merkle",
			domain_signatures("crypto::merkle", [merkle]),
			merkle_hash,
			"crypto_merkle.json",
		),
		root,
	)
	outputs[root / "src/rs/ml/matrix/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			"crate::ml::matrix",
			domain_signatures("ml::matrix", [activation]),
			activation_hash,
			"ml_activation.json",
		),
		root,
	)
	pqc_signatures = domain_signatures("crypto::pqc", [cryptography_pqc])
	pqc_root = "crate::crypto::pqc"
	pqc_imports = {"MlDsaParameters": pqc_root}
	device_names = {op["name"] for op in cryptography_pqc["device_api"]["operations"]} | {
		"mldsa_verify_hash_message", "mldsa_verify_prehashed"}
	outputs[root / "src/rs/crypto/pqc/device/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			f"{pqc_root}::device",
			[s for s in pqc_signatures if s["name"] in device_names],
			cryptography_pqc_hash,
			"crypto_pqc.json",
			pqc_imports,
		),
		root,
	)
	outputs[root / "src/rs/crypto/pqc/parity.gen.rs"] = format_rust(
		generate_signature_parity(
			f"{pqc_root}::verification",
			[s for s in pqc_signatures if s["name"] not in device_names],
			cryptography_pqc_hash,
			"crypto_pqc.json",
			pqc_imports,
		),
		root,
	)
	annex_signatures = domain_signatures("sessions", [sessions, codecs, ml_parity])
	for parity_domain in ("video", "audio", "data", "ml", "image", "runtime", "net", "media"):
		domain_sigs = [s for s in annex_signatures
					   if s.get("target", "").startswith(f"crate::{parity_domain}::")]
		if not domain_sigs:
			continue
		domain_paths = dict(SESSION_TYPE_PATHS)
		if parity_domain == "ml":
			domain_paths.update(ml_parity.get("type_paths") or {})
		outputs[root / f"src/rs/{parity_domain}/parity.gen.rs"] = format_rust(
			generate_signature_parity(
				f"crate::{parity_domain}",
				domain_sigs,
				hashlib.sha256((sessions_hash + codecs_hash + ml_parity_hash).encode()).hexdigest(),
				"sessions.json,codecs.json,ml_parity.json",
				domain_paths,
			),
			root,
		)
	for kernel in cryptography["kernels"]:
		if "poseidon2" in kernel:
			outputs[root / kernel["source"]] = generate_poseidon2_shader(kernel, cryptography, cryptography_hash)
		elif "digest" in kernel:
			outputs[root / kernel["source"]] = generate_fixed_hash_shader(kernel, cryptography_hash)
		elif kernel["name"] in {"merkle_proof_gather", "merkle_proof_verify"}:
			outputs[root / kernel["source"]] = generate_merkle_proof_shader(kernel, cryptography_hash, cryptography["proof_layout"])
	outputs[root / "src/slang/core/math/poseidon2.slang"] = generate_poseidon2_constants(cryptography, cryptography_hash)
	outputs.update(pqc_device_outputs(root, cryptography_pqc, cryptography_pqc_hash))
	outputs[root / "src/rs/crypto/pqc/verify.gen.rs"] = format_rust(generate_cryptography_api(cryptography_pqc, cryptography_pqc_hash, "verify"), root)
	outputs[root / "src/rs/vision/detection.gen.rs"] = format_rust(generate_vision_api(vision, vision_hash), root)
	for category in ("signal", "transform"):
		outputs[root / f"src/rs/audio/{category}.gen.rs"] = format_rust(generate_audio_api(audio, audio_hash, category), root)
	outputs[root / "src/rs/audio/audio.gen.rs"] = format_rust(
		generate_domain_methods("audio", domain_signatures("audio", [audio]), audio_hash, "audio.json"), root
	)
	for category in IMAGE_CATEGORIES:
		outputs[root / f"src/rs/image/{category}.gen.rs"] = format_rust(generate_image_api(image, image_hash, category), root)
	outputs[root / "src/rs/image/image.gen.rs"] = format_rust(
		generate_domain_methods("image", domain_signatures("image", [image]), image_hash, "image.json"), root
	)
	matrix_py_rust, matrix_py_stub, matrix_py_methods = generate_python_domain(
		"matrix",
		domain_signatures("matrix", [elementwise, blas, reduce, rng, index, view]),
		hashlib.sha256((elementwise_hash + blas_hash + reduce_hash + rng_hash + index_hash + view_hash).encode()).hexdigest(),
		"matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,matrix_rng.json,matrix_index.json,matrix_view.json",
		python_struct_results([index]),
	)
	outputs[root / "src/py/pyo3/matrix/matrix.gen.rs"] = format_rust(matrix_py_rust, root)
	outputs[root / "src/py/oa/_gen/matrix_native.pyi"] = matrix_py_stub
	audio_py_rust, audio_py_stub, audio_py_methods = generate_python_domain(
		"audio", domain_signatures("audio", [audio]), audio_hash, "audio.json"
	)
	outputs[root / "src/py/pyo3/audio/audio.gen.rs"] = format_rust(audio_py_rust, root)
	outputs[root / "src/py/oa/_gen/audio_native.pyi"] = audio_py_stub
	image_py_rust, image_py_stub, image_py_methods = generate_python_domain(
		"image", domain_signatures("image", [image]), image_hash, "image.json"
	)
	outputs[root / "src/py/pyo3/image/image.gen.rs"] = format_rust(image_py_rust, root)
	outputs[root / "src/py/oa/_gen/image_native.pyi"] = image_py_stub
	vision_py_rust, vision_py_stub, _ = generate_python_domain(
		"vision",
		domain_signatures("vision", [vision]),
		vision_hash,
		"vision_detection.json",
		python_struct_results([vision]),
	)
	outputs[root / "src/py/pyo3/vision/vision.gen.rs"] = format_rust(vision_py_rust, root)
	outputs[root / "src/py/oa/_gen/vision_native.pyi"] = vision_py_stub
	outputs[root / "src/py/oa/_gen/__init__.py"] = (
		banner(
			hashlib.sha256((elementwise_hash + blas_hash + reduce_hash + audio_hash + image_hash + vision_hash).encode()).hexdigest(),
			"#",
			"matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,audio.json,image.json,vision_detection.json",
		).replace("# schema=", "# schemas=")
		+ '"""Generated binding fragments; handwritten facades import from this package."""\n'
	)
	outputs[root / "src/slang/crypto/pqc/mldsa/mldsa_prehash_layout.slang"] = generate_mldsa_prehash_layout_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/crypto/pqc/mldsa/mldsa_prehash_digest.slang"] = generate_mldsa_prehash_digest_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/crypto/pqc/mldsa/mldsa_sign_workspace_layout.slang"] = generate_mldsa_sign_workspace_layout_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/crypto/pqc/mldsa/mldsa_prehash.slang"] = generate_mldsa_prehash_shader(cryptography_pqc, cryptography_pqc_hash)
	for kernel in cryptography_pqc["kernels"]:
		outputs[root / kernel["source"]] = generate_mldsa_verify_shader(kernel, cryptography_pqc_hash)
	for kernel in cryptography_pqc.get("private_kernels", []):
		outputs[root / kernel["source"]] = (generate_mlkem_keygen_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] == "ml_kem_keygen" else generate_mldsa_keygen_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] == "ml_dsa_keygen" else generate_mldsa_sign_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] in ("ml_dsa_sign", "ml_dsa_sign_message", "ml_dsa_sign_prehashed", "ml_dsa_sign_hash_message") else generate_mlkem_shared_derive_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] == "ml_kem_shared_derive" else generate_mlkem_kem_shader(kernel, cryptography_pqc_hash))
	autograd_families = sorted(
		{
			operation["autograd"]["family"]
			for operation in ml["operations"]
			if operation.get("autograd", {}).get("mode") == "generated"
		}
	)
	for family in autograd_families:
		outputs[root / f"src/rs/ml/autograd/{family}.gen.rs"] = format_rust(
			generate_ml_autograd_attachments(ml, ml_hash, family), root
		)
		outputs[root / f"src/rs/ml/autograd/node/{family}.gen.rs"] = format_rust(
			generate_ml_autograd_records(ml, ml_hash, family), root
		)
	outputs[root / "src/rs/ml/autograd/node/operation.gen.rs"] = format_rust(
		generate_ml_autograd_routing(ml, ml_hash), root
	)
	for operation in elementwise["operations"]:
		if operation.get("lowering_only", False):
			continue
		for variant in operation_variants(elementwise, operation):
			outputs[root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"] = generate_shader(
				operation,
				variant,
				elementwise,
				elementwise_hash,
			)
	for operation in blas["operations"]:
		outputs[root / f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"] = generate_blas_shader(
			operation, blas, blas_hash
		)
	for candidate in blas.get("candidate_kernels", []):
		outputs[root / f"src/slang/matrix/blas/{candidate['source_stem']}.gen.slang"] = generate_blas_candidate_shader(
			candidate, blas, blas_hash
		)
	for domain, domain_schemas, domain_hashes, domain_names in (
		("matrix", [elementwise, blas, reduce, rng, index, view],
		 [elementwise_hash, blas_hash, reduce_hash, rng_hash, index_hash, view_hash],
		 "matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,matrix_rng.json,matrix_index.json,matrix_view.json"),
		("audio", [audio], [audio_hash], "audio.json"),
		("image", [image], [image_hash], "image.json"),
		("vision", [vision], [vision_hash], "vision_detection.json"),
	):
		domain_ops = [s["name"] for s in domain_signatures(domain, domain_schemas)
					  if s["visibility"] == "pub"]
		fragment = banner(
			hashlib.sha256("".join(domain_hashes).encode()).hexdigest(), "#", domain_names)
		fragment += f"from .._native import (\n{''.join(f'\t{domain}_{name} as {name},\n' for name in domain_ops)})\n"
		fragment += "\n__all__ = [\n" + "".join(f'\t"{name}",\n' for name in domain_ops) + "]\n"
		outputs[root / f"src/py/oa/_gen/{domain}.py"] = fragment
	submodules = exports["submodules"]
	exports_source = banner(exports_hash, "#", "exports.json")
	exports_source += '\"\"\"Python bindings for OA\'s Rust semantic API.\"\"\"\n\n'
	exports_source += "from importlib.metadata import PackageNotFoundError, version\n\n"
	exports_source += f"from . import {', '.join(submodules)}\n"
	exports_source += "from ._native import (\n" + "".join(
		f"\t{name},\n" for name in exports["native_classes"]) + ")\n\n"
	exports_source += "__all__ = [\n" + "".join(
		f'\t"{name}",\n' for name in exports["native_classes"] + submodules) + "]\n"
	exports_source += '''
try:
	__version__ = version("oapython")
except PackageNotFoundError:
	# Source-tree contract tests load the built extension without installing a
	# wheel, so distribution metadata is intentionally absent.
	__version__ = "0+unknown"
'''
	outputs[root / "src/py/oa/__init__.py"] = exports_source
	merge_python_fragments(
		outputs,
		root,
		[elementwise_hash, blas_hash, reduce_hash, audio_hash, image_hash, vision_hash, cryptography_hash, bigint_hash, ntt_hash, merkle_hash, cryptography_pqc_hash],
		"matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,audio.json,image.json,vision_detection.json,crypto_hash.json,crypto_bigint.json,crypto_ntt.json,crypto_merkle.json,crypto_pqc.json",
	)
	outputs[root / "src/py/oa/_native.pyi"] = generate_native_stub(
		(FN_ROOT / "template/python/native.pyi.in").read_text(),
		{"matrix": matrix_py_methods, "audio": audio_py_methods, "image": image_py_methods},
		outputs[root / "src/py/oa/_gen/native.pyi"],
	)
	return outputs
