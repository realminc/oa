#!/usr/bin/env python3
"""Contract tests for OA's CPU and Vulkan cryptography Python surface."""

from __future__ import annotations

import array
import hashlib
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_dev_paths() -> None:
	candidates = []
	if os.getenv("OA_PYTHON_BUILD_DIR"):
		candidates.append(Path(os.environ["OA_PYTHON_BUILD_DIR"]).expanduser())
	candidates += [
		REPO_ROOT / "Build" / "Release",
		REPO_ROOT / "Build" / "Debug",
		REPO_ROOT / "build",
		REPO_ROOT / "Source" / "Python",
	]
	for path in candidates:
		if path.exists() and str(path) not in sys.path:
			sys.path.insert(0, str(path))


_add_dev_paths()
oa = pytest.importorskip("oa", reason="oa Python package is not importable")
crypto = pytest.importorskip("oa.crypto", reason="OA crypto bindings are unavailable")
if not crypto.available:
	pytest.skip("OA was built without its optional crypto backend", allow_module_level=True)
core = oa.core
runtime = oa.runtime


@pytest.fixture(scope="session")
def engine():
	if not runtime.OaInitComputeEngine():
		pytest.skip("OA compute engine could not initialize")
	yield
	shutdown = getattr(runtime, "OaShutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


def test_hash_strict_parse_and_roundtrip():
	raw = bytes(range(32))
	digest = crypto.OaHash(raw)
	assert digest.Bytes() == raw
	assert crypto.OaHash.FromHex(raw.hex()) == digest
	with pytest.raises(RuntimeError):
		crypto.OaHash.FromHex("00")
	with pytest.raises(RuntimeError):
		crypto.OaHash.FromHex("g" * 64)


def test_shake_matches_python_hashlib():
	data = b"OA Vulkan cryptography"
	assert crypto.Shake256Bytes(data, 96) == hashlib.shake_256(data).digest(96)
	assert crypto.Shake128Bytes(data, 96) == hashlib.shake_128(data).digest(96)


def test_incremental_hasher_contract():
	hasher = crypto.OaHasher()
	hasher.Update(b"OA ")
	hasher.Update(b"Crypto")
	first = hasher.Finalize()
	assert first.Bytes() == hashlib.shake_256(b"OA Crypto").digest(32)
	assert hasher.Finalize() == first
	with pytest.raises(RuntimeError):
		hasher.Update(b"late")
	hasher.Reset()
	hasher.Update(b"OA Crypto")
	assert hasher.Finalize() == first


def test_mldsa_roundtrip_and_serialized_public_types():
	keypair = crypto.GenerateKeypair()
	message = b"post-quantum OA"
	signature = crypto.Sign(message, keypair.Secret)
	assert crypto.Verify(message, signature, keypair.Pubkey)
	assert not crypto.Verify(message + b"!", signature, keypair.Pubkey)
	assert crypto.Verify(message, crypto.OaSignature(signature.Bytes()),
						 crypto.OaPublicKey(keypair.Pubkey.Bytes()))
	with pytest.raises(RuntimeError):
		crypto.OaPublicKey(b"short")
	with pytest.raises(RuntimeError):
		crypto.OaSignature(b"short")


def test_gpu_shake_matches_hashlib(engine):
	rows = [b"alpha", b"bravo", b"crypt"]
	flat = array.array("B", b"".join(rows))
	messages = core.FromBytes(flat, len(rows), len(rows[0]), core.OaScalarType.UInt8)
	with oa.Context():
		digests = crypto.Shake256(messages, 32)
	host = bytes(core.CopyToHost(digests))
	expected = b"".join(hashlib.shake_256(row).digest(32) for row in rows)
	assert host == expected


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
