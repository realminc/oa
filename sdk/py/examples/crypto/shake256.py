#!/usr/bin/env python3
"""Hash a message batch on the GPU and compare it with hashlib."""

# OA_DOC_BEGIN: crypto-shake256
import hashlib

from oa import (
	FnHash,
	FnMatrix,
	initComputeEngine,
	ScalarType,
	shutdownComputeEngine,
)


if not initComputeEngine():
	raise RuntimeError("OA could not create a Vulkan compute engine")

try:
	rows = [b"alpha", b"bravo", b"crypt"]
	inputMatrix = FnMatrix.fromBytes(
		list(b"".join(rows)),
		[3, 5],
		ScalarType.UInt8,
	)
	digests = FnHash.shake256(inputMatrix, 32)
	gpu = bytes(FnMatrix.copyToHost(digests))
	cpu = b"".join(hashlib.shake_256(row).digest(32) for row in rows)
	assert gpu == cpu
	print("3 GPU SHAKE-256 digests match the CPU oracle")
finally:
	shutdownComputeEngine()
# OA_DOC_END: crypto-shake256
