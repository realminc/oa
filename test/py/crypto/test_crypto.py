#!/usr/bin/env python3
"""Tests for OA's CPU and Vulkan cryptography Python surface."""

from __future__ import annotations

import array
import hashlib
import sys

import pytest


import oa_python_test  # noqa: F401 - bootstraps source builds

import oa

pytestmark = pytest.mark.oa_crypto
crypto = oa
core = oa.FnMatrix
runtime = oa


@pytest.fixture(scope="session")
def engine():
	if not runtime.initComputeEngine():
		pytest.fail("crypto GPU profile requires an initialized OA Vulkan engine")
	yield
	shutdown = getattr(runtime, "shutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


def test_hash_strict_parse_and_roundtrip():
	raw = bytes(range(32))
	digest = crypto.Hash(raw)
	assert digest.bytes() == raw
	assert crypto.Hash.fromHex(raw.hex()) == digest
	with pytest.raises(RuntimeError):
		crypto.Hash.fromHex("00")
	with pytest.raises(RuntimeError):
		crypto.Hash.fromHex("g" * 64)


def test_shake_matches_python_hashlib():
	data = b"OA Vulkan cryptography"
	assert crypto.shake256(data, 96) == hashlib.shake_256(data).digest(96)
	assert crypto.shake128(data, 96) == hashlib.shake_128(data).digest(96)


def test_incremental_hasher_contract():
	hasher = crypto.Hasher()
	hasher.update(b"OA ")
	hasher.update(b"Crypto")
	first = hasher.finalize()
	assert first.bytes() == hashlib.shake_256(b"OA Crypto").digest(32)
	assert hasher.finalize() == first
	with pytest.raises(RuntimeError):
		hasher.update(b"late")
	hasher.reset()
	hasher.update(b"OA Crypto")
	assert hasher.finalize() == first


def test_mldsa_roundtrip_and_serialized_public_types():
	keypair = crypto.generateKeypair()
	message = b"post-quantum OA"
	signature = crypto.sign(message, keypair.secret)
	assert crypto.verify(message, signature, keypair.pubkey)
	assert not crypto.verify(message + b"!", signature, keypair.pubkey)
	assert crypto.verify(message, crypto.Signature(signature.bytes()),
						 crypto.PublicKey(keypair.pubkey.bytes()))
	with pytest.raises(RuntimeError):
		crypto.PublicKey(b"short")
	with pytest.raises(RuntimeError):
		crypto.Signature(b"short")


def test_gpu_shake_matches_hashlib(engine):
	rows = [b"alpha", b"bravo", b"crypt"]
	flat = array.array("B", b"".join(rows))
	messages = core.fromBytes(flat, len(rows), len(rows[0]), oa.ScalarType.UInt8)
	digests = oa.FnHash.shake256(messages, 32)
	host = bytes(core.copyToHost(digests))
	expected = b"".join(hashlib.shake_256(row).digest(32) for row in rows)
	assert host == expected


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
