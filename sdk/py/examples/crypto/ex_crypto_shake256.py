# OA_DOC_BEGIN: crypto-shake256
import hashlib

import oa

engine = oa.Engine()

rows = [b"alpha", b"bravo", b"crypt"]
messages = oa.Matrix.from_u8(
	engine,
	[3, 5],
	list(b"".join(rows)),
)

digests = oa.cryptography.hash_shake256(messages, 32)

gpu = bytes(digests.read_u8())
cpu = b"".join(hashlib.shake_256(row).digest(32) for row in rows)
assert gpu == cpu

print("3 GPU SHAKE-256 digests match the CPU oracle")
# OA_DOC_END: crypto-shake256
