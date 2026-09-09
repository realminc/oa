# OA Rust Crypto

**Status:** Experimental CPU primitive checkpoint

**OA donor:** `source/cpp/include/oa/crypto/{keccak,hash}.h` and
`source/cpp/lib/oa/crypto/{keccak,hash}.cpp`

## Implemented contract

The Rust module directly ports OA's language-neutral CPU algorithms:

- the 24-round Keccak-f[1600] permutation with little-endian lanes;
- one-shot and incremental SHAKE-128/256;
- KMAC-256 with SP 800-185 left/right encoding and byte padding;
- the strict typed 32-byte SHAKE-256 `Hash` value;
- idempotent-finalize/reset incremental `Hasher`;
- arbitrary-leaf Merkle root, materialized tree, checked proof construction,
  and fail-closed proof verification.

Incremental SHAKE rejects absorption after squeezing until reset. KMAC uses
checked byte/bit-length arithmetic. Sponge state, buffered input, KMAC prefix,
and key-encoding temporaries are erased through `core::memory::zero_secure`.
This bounds observable host-memory erasure; it is not a claim that compiler,
register, cache, swap, or whole-system copies cannot exist.

## Evidence

The external CPU suite covers the Keccak zero-state reference permutation,
FIPS 202 SHAKE-128/256 known-answer vectors, incremental absorb/squeeze across
rate boundaries, the NIST SP 800-185 KMAC-256 sample-four vector, strict
hash parsing, hasher lifecycle, odd Merkle trees, every-leaf proofs, and
negative proof/index paths.

```bash
cargo test --test crypto
```

This is engineering evidence, not FIPS 140 validation, certification, or a
system-wide constant-time claim.

## Planned

OA's Vulkan batch SHAKE, Keccak, and power-of-two Merkle routes remain Planned
until U8 Matrix schemas, donor Slang kernels, executable chaining, CPU
differential tests, and Vulkan validation land. ML-DSA-65 remains Planned until
the Rust dependency, typed secret ownership, serialization ABI, negative tests,
and side-channel scope are explicitly qualified. Secret keys must not enter
generic Matrix storage.
