# vkPQC roadmap

**Status:** Planned research

**Updated:** 2026-09-10

**Subsystem contract:** [OA Rust Cryptography](../../cryptography/oaCryptography.md)

**GPU-secret security:**
[GPU secret execution and observability](../../cryptography/oaGpuSecretSecurity.md)

## 1. Objective

`vkPQC` is the working name for a vendor-neutral Vulkan backend for selected
post-quantum cryptography. It is not a second public OA namespace. Public
ownership remains under `oa::cryptography::pqc`; Rust owns types, validation,
graph lowering, lifetime, and evidence, while Slang owns device algorithms
compiled to SPIR-V.

The first target is not complete NVIDIA cuPQC SDK parity. It is a smaller
`vkPQC-PK` slice:

1. ML-DSA-65 batch verification;
2. ML-DSA-44/65/87 verification;
3. ML-KEM-512/768/1024 key generation, encapsulation, and decapsulation;
4. ML-DSA-44/65/87 key generation and signing after secret-device execution is
   independently admitted;
5. device-callable composition and measured capability-selected tuning.

Generic big integers, generic large NTTs, SHA-2, Poseidon2, and generalized
Merkle construction are separate possible cryptography projects. They are not
prerequisites for FIPS 203 or FIPS 204 and must not delay the first verified
PQC slice.

## 2. Reference scope and current gap

As of 2026-09-10, NVIDIA cuPQC 0.6 documents four device-side libraries:
BigInt, NTT, Hash, and PK. cuPQC-PK supports all three standardized ML-KEM
parameter sets with key generation, encapsulation, and decapsulation, and all
three ML-DSA parameter sets with key generation, signing, and verification.
The PK execution model places one operation in one CUDA block with selectable
32, 64, 128, or 256-thread block sizes.

The comparison below is a named-capability inventory, not a performance or
security-equivalence claim.

| Area | cuPQC 0.6 reference | OARS current checkpoint | Planned gap |
|---|---|---|---|
| BigInt | fixed 32–4096-bit device arithmetic, thread and warp-cooperative modes | none | entire optional library |
| Generic NTT | custom fields below `2^62`, power-of-two lengths `2^2`–`2^24`, standard and staged execution | none | entire optional library |
| Hash | SHA-2, SHA-3, SHAKE, Poseidon2, single/multi-block Merkle generation and proofs | Vulkan Keccak-f[1600], SHAKE-128/256, and SHAKE-256 power-of-two root reduction | SHA-3 required for ML-KEM; remaining breadth optional |
| ML-KEM | 512/768/1024 × keygen/encaps/decaps | none | 9 of 9 device parameter-operation combinations |
| ML-DSA | 44/65/87 × keygen/sign/verify | CPU ML-DSA-65 only | 9 of 9 device parameter-operation combinations |
| Device composition | device functions callable from surrounding CUDA kernels | standalone Vulkan operation dispatch | reusable Slang device routines plus OA entry-point wrappers |
| Secret execution | entropy and workspace contracts; implementation reports side-channel work | host-only `SecureBuffer`; no GPU secret type | ownership, entropy, erasure, observability, and review contract |

By named GPU capability, OARS implements none of the 18 cuPQC-PK
parameter-operation combinations. It nevertheless has reusable enabling work:

- a verified `uint2` Keccak-f[1600] Slang implementation and Vulkan
  SHAKE-128/256;
- U8 Matrix storage and checked packed-byte allocation;
- schema-generated semantic operations and physical shader identities;
- deferred executable graphs, explicit events, barriers, and hardware tests;
- CPU ML-DSA-65 and independent hash oracles;
- typed keys and signatures plus a host secret-erasure boundary.

The resulting planning estimate is 20–30 percent of the infrastructure needed
for the first ML-DSA-65 verifier, 10–15 percent readiness for the complete PK
target, and less than 10 percent of full cuPQC 0.6 breadth. These are rough
engineering-readiness estimates. They are not line-count, completion, schedule,
or performance percentages.

## 3. Donor disposition

OA C++ once contained ten experimental ML-DSA, NTT, and polynomial Slang
files, approximately 877 source lines in the last retained Git objects. They
were removed in OA v0.6.106 because they lacked complete algorithms, consumers,
oracles, manifest authority, and security evidence.

The historical ML-DSA verifier explicitly returned invalid for every input.
Key generation and signing were WIP status writers rather than cryptographic
implementations. The fixed constants, layouts, and one-workgroup design notes
may be independently checked against FIPS 204, but historical objects are not
an implementation baseline and must never be restored wholesale.

cuPQC is a systems and benchmark reference, not donor source. Its SDK license
restricts SDK use to NVIDIA GPU systems and prohibits reverse engineering and
ungranted derivative creation. vkPQC therefore implements the public standards
independently, using NIST material and separately licensed CPU implementations
as correctness oracles.

## 4. Architectural contract

### 4.1 Public surface

The intended structural owner is:

```text
oa::cryptography::pqc
  ml_dsa
    verify_batch
    keygen_batch     Planned after secret-execution admission
    sign_batch       Planned after secret-execution admission
  ml_kem
    keygen_batch     Planned after secret-execution admission
    encapsulate_batch
    decapsulate_batch Planned after secret-execution admission
```

Exact names remain Planned until the first operation schema and API review.
The crate root may identity-re-export admitted principal value types, but it
does not re-export stateless operations. `vkPQC` remains a backend/project name,
not a public module competing with `cryptography::pqc`.

### 4.2 Source ownership

The planned source split is:

```text
src/rs/cryptography/pqc.rs                public PQC facade
src/rs/cryptography/pqc/
  ml_dsa.rs                               public typed ML-DSA operations
  ml_kem.rs                               public typed ML-KEM operations
  value.rs                                algorithm-specific public values
  lowering.rs                             private semantic-to-device lowering
  secret.rs                               private secret-device lifetime policy

src/slang/cryptography/pqc/
  common/
    keccak.slang                          shared verified sponge implementation
    field_mldsa.slang                     q = 8380417 arithmetic
    field_mlkem.slang                     q = 3329 arithmetic
    ntt_mldsa.slang                       fixed 256-coefficient ML-DSA transform
    ntt_mlkem.slang                       fixed ML-KEM transform
    packing.slang                         canonical encode/decode helpers
  ml_dsa/
    verify.slang
    keygen.slang
    sign.slang
  ml_kem/
    keygen.slang
    encapsulate.slang
    decapsulate.slang
```

Files are introduced only with a consuming vertical slice. A directory tree is
not evidence of capability. Shared Slang routines remain private build inputs;
schema-owned wrapper entry points provide stable semantic and physical
identities.

### 4.3 Values and storage

Public and secret material must not be collapsed into one generic byte Matrix:

| Value | Secrecy | Storage rule |
|---|---|---|
| message batch | caller-defined, potentially confidential | packed byte storage plus checked offsets/lengths; no fixed-row padding assumption |
| public key batch | public | typed value may use ordinary device backing |
| signature batch | public after creation | typed value; malformed and non-canonical encodings fail closed |
| ciphertext batch | public | typed ML-KEM value |
| shared-secret batch | secret | private non-mappable/non-exportable device owner; never generic Matrix |
| secret-key batch | secret | private non-mappable/non-exportable device owner; never generic Matrix |
| entropy batch | secret and single-use | explicit seed/entropy input, consumed once and erased after its final event |
| workspace | secret if any content depends on a secret | inherits the strictest secrecy of the operation using it |

Uniform Matrix rows are insufficient for full message semantics. The admitted
batch contract must support a packed byte buffer with overflow-checked offsets
and lengths, or explicitly support only prehashed/fixed-size input. The choice
must be fixed in the schema rather than hidden in lowering.

### 4.4 Execution and completion

- One workgroup initially owns one independent PQC operation; batching occurs
  across workgroups. Candidate metadata must express exclusive output and
  secret-workspace partitions so host admission and generated overlap tests can
  reject cross-operation writes before the route is admitted.
- The portable path does not assume a 32-lane subgroup or native shader `i64`.
  Capability-selected subgroup or integer variants remain private lowering.
- Intermediate polynomials should remain in registers or workgroup memory when
  occupancy measurements justify it. Register spilling is treated as secret
  device storage, not assumed harmless.
- Verification, key generation, signing, encapsulation, and decapsulation each
  remain one semantic operation even if lowering emits several executable
  dispatches.
- Every intermediate and secret allocation remains alive through the exact
  final consumer event. Erasure is a separately recorded device write and the
  allocation cannot return to a pool until the erasure event completes.
- `Drop` does not submit or wait. An admitted secret owner therefore requires
  an explicit failure-bearing close/erase boundary or engine retirement policy
  that records erasure before reuse without relying on destruction.
- No operation silently falls back from Vulkan to CPU after accepting a
  device-secret contract.
- Exclusive partitioning proves a race-safety property only. It does not prove
  constant-time behavior, prevent register spills, or protect keys from the
  owning process, a privileged debugger, or capture tooling.

## 5. Required algorithm substrate

| Primitive | ML-DSA | ML-KEM | Current Vulkan status |
|---|:---:|:---:|---|
| Keccak-f[1600] | yes | yes | implemented |
| SHAKE-128/256 | yes | yes | implemented as batch operations; must become safely reusable inside PQC shaders |
| SHA3-256/512 | no | yes | missing; reuse the verified permutation with FIPS 202 domain separation |
| fixed polynomial NTT/INTT | yes | yes | missing; two algorithm-specific modulus/layout contracts |
| pointwise and matrix-vector polynomial arithmetic | yes | yes | missing |
| rejection and noise sampling | yes | yes | missing |
| canonical key/signature/ciphertext packing | yes | yes | missing |
| rounding, decomposition, hints, high/low bits | yes | no | missing |
| challenge `SampleInBall` | yes | no | missing |
| compression/decompression and implicit rejection | no | yes | missing |

Generic cuPQC-NTT is deliberately excluded from this dependency chain. A
verified fixed transform for each FIPS algorithm is smaller, easier to test,
and does not falsely claim arbitrary-field or `2^24` transform support.

## 6. Dependency-ordered implementation

### Phase 0 — freeze standards and evidence

1. Pin the exact FIPS 203 and FIPS 204 revisions plus their current errata.
2. Record every parameter, byte layout, external/internal interface, context,
   prehash, deterministic/randomized mode, and malformed-input outcome.
3. Add deterministic CPU component oracles for both moduli, NTTs, sampling,
   packing, rounding, hints, and implicit rejection.
4. Import licensed NIST ACVP/KAT material with source and revision provenance.
5. Define schema shapes for packed messages and typed batch values.

Acceptance: all component oracles run without Vulkan and reproduce official or
independent reference vectors.

### Phase 1 — common public-data device primitives

1. Make Keccak/SHAKE callable from a composed Slang module without duplicating
   the permutation.
2. Add SHA3-256/512 wrappers and FIPS 202 vectors.
3. Implement ML-DSA modular arithmetic, fixed NTT/INTT, pointwise multiply,
   packing, rounding, hints, uniform sampling, and `SampleInBall`.
4. Test each primitive against its CPU oracle across randomized, boundary,
   odd-batch, poison, reuse, and malformed cases.

Acceptance: NTT round trips and polynomial multiplication pass property tests;
all shader modules pass Slang compilation, reflected ABI checks, `spirv-val`,
core validation, synchronization validation, and GPU-assisted bounds testing.

### Phase 2 — ML-DSA-65 batch verification

1. Decode public keys and signatures with canonical-encoding rejection.
2. Check norm and hint constraints without accepting malformed encodings.
3. Expand the public matrix, reconstruct the challenge, perform polynomial
   verification, and compare the final challenge.
4. Return one explicit validity result per batch element; no error turns an
   invalid signature into success.
5. Differentially test valid, tampered, wrong-key, truncated, non-canonical,
   and randomized cases against the CPU oracle.

Acceptance: public-only ML-DSA-65 verification is the first possible vkPQC
Experimental capability. It does not admit GPU signing or key generation.

### Phase 3 — all ML-DSA verification parameter sets

Parameterize the proven verifier for ML-DSA-44 and ML-DSA-87. Generated
contracts own sizes and algorithm identities; implementation sharing must not
erase distinct public types or accept cross-parameter encodings.

Acceptance: all parameter sets pass NIST vectors and randomized differential
tests on every device class claimed by the capability report.

### Phase 4 — ML-KEM public algorithm proof

Implement q=3329 arithmetic, its transform and representation, sampling,
compression, serialization, encryption/decryption core, and implicit rejection.
Prove deterministic component behavior before connecting external entropy or
secret-device storage.

Acceptance: deterministic internal key generation, encapsulation, and
decapsulation reproduce NIST component vectors on public test data. This phase
is not yet a production secret-bearing API.

### Phase 5 — secret-device execution boundary

Implement the requirements in
[GPU secret execution and observability](../../cryptography/oaGpuSecretSecurity.md):

1. non-mappable, non-exportable typed secret allocations;
2. explicit host CSPRNG and deterministic-test entropy interfaces;
3. event-bounded single-use entropy and workspace retirement;
4. device erasure before allocation reuse;
5. graph, log, diagnostic, panic, crash-dump, and capture redaction policy;
6. development, qualification, ordinary-production, and confidential-compute
   profiles;
7. side-channel review and capability evidence.

Acceptance: this phase admits no algorithm by itself. It enables later
secret-bearing operations to request review.

### Phase 6 — ML-KEM and ML-DSA secret operations

1. Promote ML-KEM-512/768/1024 keygen, encapsulation, and decapsulation through
   the secret-device boundary.
2. Implement ML-DSA deterministic and randomized signing, rejection loops,
   key generation, contexts, and any admitted prehash interface.
3. Extend ML-DSA to all three parameter sets only after ML-DSA-65 passes the
   long-rejection, malformed, entropy, erasure, and leakage gates.

Signing must not impose an unproved rejection limit. Current NIST ACVP ML-DSA
tests include long-rejection cases because premature abort behavior can leak
secret information.

Acceptance: every secret-bearing operation has an explicit entropy source,
typed secret storage, exact erasure completion, and independent security review.

### Phase 7 — composition and performance qualification

1. Expose private callable Slang routines for fusion inside selected OA kernels
   while retaining schema-owned standalone wrappers.
2. Measure 32/64/128/256-thread workgroup candidates where device limits permit;
   do not expose workgroup size publicly.
3. Add private variants for subgroup capabilities, native/non-native 64-bit
   arithmetic, shared-memory capacity, and register pressure.
4. Compare Vulkan and cuPQC CUDA on the same NVIDIA GPU, inputs, batch sizes,
   transfer boundaries, warmup, clocks, and correctness oracle.
5. Qualify AMD, Intel, NVIDIA, and other claimed Vulkan devices separately.

No result is called cuPQC parity merely because both APIs complete. Performance
comparison requires identical operation semantics and records pipeline setup,
uploads, submission, synchronization, device execution, download, validation,
tooling, driver, clocks, temperature, and fallback evidence.

### Phase 8 — optional general cryptography SDK breadth

Only a demonstrated OA consumer may initiate any of:

- generic fixed-width device BigInt;
- generic arbitrary-prime cyclic NTT;
- SHA-2 and SHA-3 family breadth beyond PQC requirements;
- Poseidon2 parameter sets;
- generalized single/multi-block Merkle generation and proof operations.

Each is an independent operation family with its own schema, oracle, security
claims, and benchmark. Implementing them does not retroactively strengthen the
PQC security claim.

## 7. Verification and security gates

Every promoted algorithm requires:

- official NIST ACVP/KAT coverage for every admitted parameter set and mode;
- randomized differential testing against at least one separately maintained
  CPU implementation;
- canonical-decoding and malformed-input fuzzing;
- zero, one, odd, large practical, mixed-length, alias, reuse, and poison tests;
- fixed-seed fresh-process determinism evidence where deterministic behavior is
  claimed;
- ASAN/UBSAN for host and FFI boundaries;
- SPIR-V validation plus separate Vulkan core, synchronization, and GPU-assisted
  validation on named devices;
- timing, memory-access, divergence, cache, and register-spill analysis for
  secret-bearing paths;
- an independent cryptographic implementation review before any Shipped or
  production-security claim.

FIPS algorithm conformance is not FIPS 140 module validation. Passing vectors,
using an approved algorithm, or running in confidential-compute hardware does
not authorize a certification claim.

## 8. Rejected shortcuts

- Restoring the historical always-invalid or status-only OA shaders.
- Translating proprietary cuPQC binaries or implementation material.
- Treating generic Matrix as secret-key, entropy, or shared-secret storage.
- Claiming device-local memory prevents capture or privileged observation.
- Claiming erasure at `Drop` when no completed device overwrite exists.
- Using Vulkan protected memory without a valid declassification/output design.
- Publishing signing/keygen before verification and public-data primitives are
  independently proven.
- Assuming NVIDIA subgroup, integer, shared-memory, or occupancy behavior on
  other Vulkan devices.
- Comparing a Vulkan kernel timestamp against a CUDA end-to-end measurement.
- Describing vector parity as constant-time, side-channel-safe, certified, or
  production-ready.

## 9. Primary references

- [NIST FIPS 203 — ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 — ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
- [NIST ACVP documentation](https://pages.nist.gov/ACVP/)
- [NVIDIA cuPQC SDK](https://docs.nvidia.com/cuda/cupqc/)
- [cuPQC-PK operators](https://docs.nvidia.com/cuda/cupqc/api/cupqc_pk/operators.html)
- [cuPQC security notes](https://docs.nvidia.com/cuda/cupqc/additional/security_notes.html)
- [cuPQC SDK license](https://docs.nvidia.com/cuda/cupqc/additional/license.html)
- [Vulkan protected-memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html#memory-protected-memory)
