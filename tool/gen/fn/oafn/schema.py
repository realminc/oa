"""Schema registry: name, flag, default path and validator per schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

from .audio import validate_audio_schema
from .bigint import validate_bigint_schema
from .hash import validate_cryptography_hash_schema
from .image import validate_image_schema
from .matrix import (
	validate_blas_schema,
	validate_index_schema,
	validate_matrix_autograd_schema,
	validate_reduce_schema,
	validate_rng_schema,
	validate_schema,
	validate_view_schema,
)
from .merkle import validate_merkle_schema
from .misc import validate_exports_schema, validate_sessions_schema
from .ml import validate_activation_schema, validate_ml_schema
from .ntt import validate_ntt_schema
from .paths import REPO_ROOT
from .pqc import validate_cryptography_pqc_schema
from .types import validate_types_schema
from .util import SchemaError
from .vision import validate_vision_schema

Schemas = dict[str, tuple[dict[str, Any], str]]


class SchemaSpec(NamedTuple):
	flag: str
	path: Path
	validator: Any


SCHEMAS: dict[str, SchemaSpec] = {
	"elemwise": SchemaSpec("--schema", Path("tool/gen/fn/schema/matrix/matrix_elemwise.json"), validate_schema),
	"blas": SchemaSpec("--blas-schema", Path("tool/gen/fn/schema/matrix/matrix_blas.json"), validate_blas_schema),
	"reduce": SchemaSpec("--reduce-schema", Path("tool/gen/fn/schema/matrix/matrix_reduce.json"), validate_reduce_schema),
	"rng": SchemaSpec("--rng-schema", Path("tool/gen/fn/schema/matrix/matrix_rng.json"), validate_rng_schema),
	"index": SchemaSpec("--index-schema", Path("tool/gen/fn/schema/matrix/matrix_index.json"), validate_index_schema),
	"view": SchemaSpec("--view-schema", Path("tool/gen/fn/schema/matrix/matrix_view.json"), validate_view_schema),
	"ml": SchemaSpec("--ml-schema", Path("tool/gen/fn/schema/ml/ml_training.json"), validate_ml_schema),
	"ml_activation": SchemaSpec("--ml-activation-schema", Path("tool/gen/fn/schema/ml/ml_activation.json"), validate_activation_schema),
	"audio": SchemaSpec("--audio-schema", Path("tool/gen/fn/schema/audio/audio.json"), validate_audio_schema),
	"hash": SchemaSpec("--crypto-hash-schema", Path("tool/gen/fn/schema/crypto/crypto_hash.json"), validate_cryptography_hash_schema),
	"merkle": SchemaSpec("--crypto-merkle-schema", Path("tool/gen/fn/schema/crypto/crypto_merkle.json"), validate_merkle_schema),
	"ntt": SchemaSpec("--crypto-ntt-schema", Path("tool/gen/fn/schema/crypto/crypto_ntt.json"), validate_ntt_schema),
	"pqc": SchemaSpec("--crypto-pqc-schema", Path("tool/gen/fn/schema/crypto/crypto_pqc.json"), validate_cryptography_pqc_schema),
	"bigint": SchemaSpec("--crypto-bigint-schema", Path("tool/gen/fn/schema/crypto/crypto_bigint.json"), validate_bigint_schema),
	"image": SchemaSpec("--image-schema", Path("tool/gen/fn/schema/image/image.json"), validate_image_schema),
	"vision": SchemaSpec("--vision-schema", Path("tool/gen/fn/schema/vision/vision_detection.json"), validate_vision_schema),
	"sessions": SchemaSpec("--sessions-schema", Path("tool/gen/fn/schema/sessions.json"), validate_sessions_schema),
	"codecs": SchemaSpec("--codecs-schema", Path("tool/gen/fn/schema/codecs.json"), validate_sessions_schema),
	"ml_parity": SchemaSpec("--ml-parity-schema", Path("tool/gen/fn/schema/ml_parity.json"), validate_sessions_schema),
	"exports": SchemaSpec("--exports-schema", Path("tool/gen/fn/schema/exports.json"), validate_exports_schema),
	"data_types": SchemaSpec("--data-types-schema", Path("tool/gen/fn/schema/data/data_types.json"), validate_types_schema),
	"input_types": SchemaSpec("--input-types-schema", Path("tool/gen/fn/schema/input/input_types.json"), validate_types_schema),
	"ui_types": SchemaSpec("--ui-types-schema", Path("tool/gen/fn/schema/ui/ui_types.json"), validate_types_schema),
	"video_types": SchemaSpec("--video-types-schema", Path("tool/gen/fn/schema/video/video_types.json"), validate_types_schema),
	"runtime_types": SchemaSpec("--runtime-types-schema", Path("tool/gen/fn/schema/runtime/runtime_types.json"), validate_types_schema),
	"ml_types": SchemaSpec("--ml-types-schema", Path("tool/gen/fn/schema/ml/ml_types.json"), validate_types_schema),
	"audio_types": SchemaSpec("--audio-types-schema", Path("tool/gen/fn/schema/audio/audio_types.json"), validate_types_schema),
	"media_types": SchemaSpec("--media-types-schema", Path("tool/gen/fn/schema/media/media_types.json"), validate_types_schema),
	"viewer_types": SchemaSpec("--viewer-types-schema", Path("tool/gen/fn/schema/viewer/viewer_types.json"), validate_types_schema),
	"plot_types": SchemaSpec("--plot-types-schema", Path("tool/gen/fn/schema/plot/plot_types.json"), validate_types_schema),
	"net_types": SchemaSpec("--net-types-schema", Path("tool/gen/fn/schema/net/net_types.json"), validate_types_schema),
	"vision_types": SchemaSpec("--vision-types-schema", Path("tool/gen/fn/schema/vision/vision_types.json"), validate_types_schema),
	"crypto_types": SchemaSpec("--crypto-types-schema", Path("tool/gen/fn/schema/crypto/crypto_types.json"), validate_types_schema),
	"core_types": SchemaSpec("--core-types-schema", Path("tool/gen/fn/schema/core/core_types.json"), validate_types_schema),
}


def load_validated_schema(path: Path, validator: Any) -> tuple[dict[str, Any], str]:
	raw = path.read_bytes()
	try:
		schema = json.loads(raw)
	except json.JSONDecodeError as error:
		raise SchemaError(f"invalid JSON: {error}") from error
	if not isinstance(schema, dict):
		raise SchemaError("schema root must be an object")
	validator(schema)
	if schema.get("domain") == "matrix":
		validate_matrix_autograd_schema(schema)
	return schema, hashlib.sha256(raw).hexdigest()


def load_schema(name: str, path: Path | None = None, *, root: Path = REPO_ROOT) -> tuple[dict[str, Any], str]:
	spec = SCHEMAS[name]
	return load_validated_schema(path.resolve() if path else root / spec.path, spec.validator)


def load_schemas(root: Path, overrides: dict[str, Path] | None = None) -> Schemas:
	overrides = overrides or {}
	return {
		name: load_schema(name, overrides.get(name), root=root)
		for name in SCHEMAS
	}


def ensure_schemas(loaded: Schemas) -> Schemas:
	merged = dict(loaded)
	for name in SCHEMAS:
		if name not in merged:
			merged[name] = load_schema(name)
	return merged
