# OA Rust Cryptography

**Status:** Experimental complete public-donor checkpoint

**OA donor:** `source/cpp/include/oa/crypto/`,
`source/cpp/lib/oa/crypto/`, and its five shipping compute shaders

**Planned vkPQC work:**
[vkPQC roadmap](../architecture/roadmap/vkPqcRoadmap.md)

**Secret-device boundary:**
[GPU secret execution and observability](oaGpuSecretSecurity.md)

## Naming and ownership

`oa::cryptography` is the public umbrella. `crypto` is rejected as an umbrella
because it is overloaded in product language, `hash` is too narrow, and `pqc`
would falsely classify general Keccak, SHAKE, KMAC, Merkle, and secure-memory
facilities as post-quantum algorithms.

- `oa::cryptography::{shake256, Hash, ...}` owns general CPU primitives.
- `oa::cryptography::hash::{shake256, keccak_f1600, merkle_root}` owns
  schema-backed Vulkan batch operations over U8 `Matrix` values.
- `oa::cryptography::pqc` owns ML-DSA-65 keys, signatures, and operations.
- `vkPQC` is reserved as a possible backend or standalone project name. It is
  not used for this API because device-side ML-DSA is not shipped here.

The old experimental `oa::crypto` route is removed so there is one canonical
owner. Principal values are identity-re-exported as `oa::{Hash, PublicKey,
SecretKey, Signature, Keypair, SecureBuffer}` for C++/Python continuity; their
structural owners remain `cryptography` and `cryptography::pqc`. Callers may
also locally alias `oa::cryptography` when a shorter domain name is useful.

## Implemented contract

The CPU primitive surface directly ports OA's language-neutral algorithms:

- the 24-round Keccak-f[1600] permutation with little-endian lanes;
- one-shot and incremental SHAKE-128/256;
- KMAC-256 with SP 800-185 left/right encoding and byte padding;
- the strict typed 32-byte SHAKE-256 `Hash` value;
- idempotent-finalize/reset incremental `Hasher`;
- arbitrary-leaf Merkle root, materialized tree, checked proof construction,
  and fail-closed proof verification;
- a borrowed `SecureBuffer` that best-effort locks Linux pages, securely erases
  its range on reset/drop, and never assumes allocation ownership;
- typed ML-DSA-65 public keys, secret keys, signatures, key generation,
  message/hash signing, fail-closed verification, and fixed-size public
  serialization through RustCrypto `ml-dsa` 0.1.1.

Incremental SHAKE rejects absorption after squeezing until reset. KMAC uses
checked byte/bit-length arithmetic. Sponge state, buffered input, KMAC prefix,
and key-encoding temporaries are erased through `core::memory::zero_secure`.
This bounds observable host-memory erasure; it is not a claim that compiler,
register, cache, swap, or whole-system copies cannot exist.

The ML-DSA secret type is non-serializing,
non-`Clone`, redacted in `Debug`, and backed by a dependency whose seed and
expanded state implement zeroization on drop. Secret keys never enter generic
`Matrix` storage.

The Vulkan surface ports batch SHAKE-128/256, Keccak-f[1600], and power-of-two
Merkle reduction. Its operation identities and physical kernels are generated
from `tools/gen/fn/schema/cryptography_hash.json`. U8 matrices keep exact
logical shapes while their allocation is rounded to four bytes, preventing the
donor byte-address shader's final packed load from crossing a storage bound.
Merkle reduction is one semantic operation lowered to as many executable levels
as required, preserving deferred SHAKE-to-Merkle chaining.

The donor's shared 64-bit Keccak shader was replaced with its own equivalent
`uint2` implementation from `keccakF1600.slang`, then reused by SHAKE and
Merkle. This deliberate adaptation avoids requiring `shaderInt64` from every
OARS device.

## Evidence

```bash
python3 tools/gen/fn/generate.py --check
cargo test --test cryptography
cargo test --test cryptography -- --ignored
```

CPU evidence covers the prior vectors plus secure erasure, ML-DSA key/signature
sizes, round trips, tampered messages/signatures, wrong keys, hashes, and parse
length failures. Hardware Vulkan tests differentially check SHAKE across rate
boundaries and empty rows, batched Keccak, Merkle sizes 1 through 1024,
deferred chaining, and invalid rank/dtype/shape contracts. Shader build gates
compile Slang, reflect exact schema push ABIs, and run `spirv-val` for Vulkan
1.3.

This is engineering evidence, not FIPS 140 validation,
certification, or a system-wide constant-time claim.

## Deliberately not shipped

OA donor ML-DSA Vulkan experiments were incomplete and were never part of its
public signing API. They remain outside this port. A future `vkPQC` backend must
first land a complete operation schema, capability policy, independent oracle,
secret-storage contract, validation evidence, and end-to-end device ML-DSA.

The roadmap deliberately separates focused cuPQC-PK-like coverage from full
cuPQC SDK breadth. ML-DSA/ML-KEM require fixed polynomial transforms and
Keccak-family primitives; they do not require a generic 4096-bit integer
library, arbitrary-field large NTTs, Poseidon2, or generalized Merkle trees.

Ordinary Vulkan device-local memory is not a confidentiality boundary against
the owning process, capture layers, debuggers, privileged host software, or
crash artifacts. GPU verification is admitted first because it needs no secret
key, entropy, or shared-secret input and can be qualified with public test
messages. GPU key generation, signing, shared-secret handling, and decapsulation
remain Planned until typed secret-device storage, entropy, event-bounded
erasure, observability policy, and an exact deployment threat model pass the
separate secret-device contract.
