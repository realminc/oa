"""Annex schema validation (sessions, exports, codecs, parity)."""

from __future__ import annotations

from typing import Any

from .util import IDENTIFIER, require

def validate_sessions_schema(schema: dict[str, Any]) -> None:
	records = schema.get("signature_records")
	require(isinstance(records, list) and records and all(
		isinstance(r, dict) and isinstance(r.get("name"), str) and isinstance(r.get("signature"), dict)
		for r in records), "Sessions schema must carry signature_records")


SESSION_TYPE_PATHS = {
	"MediaPlayer": "crate::media", "MediaPlayerConfig": "crate::media",
	"SdlWindow": "crate::runtime", "LogComponent": "crate::runtime", "LogOptions": "crate::runtime",
	"EngineBuilder": "crate::runtime", "DeviceSummary": "crate::runtime",
	"TcpStream": "crate::net", "TcpListener": "crate::net",
	"VideoCodec": "crate::video", "VideoContainerInfo": "crate::video", "VideoDecoder": "crate::video",
	"VideoMuxerConfig": "crate::video",
	"AudioEncodeProfile": "crate::audio", "AudioEncoder": "crate::audio",
	"DataFrame": "crate::data", "DataFrameView": "crate::data", "DataLoader": "crate::data",
	"DataLoaderConfig": "crate::data", "Dataset": "crate::data", "DsOhlcv": "crate::data",
	"OhlcvView": "crate::data", "OhlcvWindowConfig": "crate::data", "OhlcvWindows": "crate::data",
	"Subset": "crate::data", "UnixTimeUnit": "crate::data",
	"EnvironmentExecution": "crate::ml::environment",
	"TradingConfig": "crate::ml::environment", "TradingEnvironment": "crate::ml::environment",
	"TradingSeries": "crate::ml::environment",
	"CheckpointManager": "crate::ml::checkpoint", "CheckpointManagerConfig": "crate::ml::checkpoint",
}


def validate_exports_schema(schema: dict[str, Any]) -> None:
	names = schema.get("native_classes")
	submodules = schema.get("submodules")
	require(isinstance(names, list) and names and all(
		isinstance(n, str) and IDENTIFIER.fullmatch(n) for n in names),
		"Exports schema must carry native_classes")
	require(isinstance(submodules, list) and submodules and all(
		isinstance(n, str) and IDENTIFIER.fullmatch(n) for n in submodules),
		"Exports schema must carry submodules")
	require(len(names) == len(set(names)) and len(submodules) == len(set(submodules)),
		"Exports schema entries must be unique")
