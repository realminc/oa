# GPU secret execution and observability

**Status:** Planned security contract

**Updated:** 2026-09-10

**Roadmap:** [vkPQC roadmap](../architecture/roadmap/vkPqcRoadmap.md)

## 1. Scope and conclusion

This document governs any OA operation that places private keys, shared
secrets, entropy, secret-derived polynomial state, or secret-dependent
workspace on a GPU.

The central constraint is absolute: an ordinary CUDA or Vulkan process cannot
make its GPU plaintext inaccessible to the same process, injected code, an
authorized debugger, an API capture layer, or a sufficiently privileged
non-confidential host. Device-local memory, register/shared-memory placement,
short lifetimes, non-mappable allocation, and explicit erasure reduce exposure;
they do not create that trust boundary.

NVIDIA's answer to an untrusted host is not cuPQC itself. It is a supported
Hopper/Blackwell Confidential Computing deployment: a confidential VM, whole
GPU pass-through, GPU and platform attestation, protected CPU-to-GPU transport,
and full CC-On mode with DevTools mode off and restricted debug/counter paths.
Even that protects
against actors outside the attested confidential workload. Code running inside
the trusted guest and the authorized application can still intentionally copy
or disclose its own plaintext.

Consequently:

- OARS may first accelerate verification because it requires no secret key,
  entropy, or shared-secret input. Its initial qualification uses public test
  messages and makes no message-confidentiality claim.
- GPU key generation, signing, ML-KEM decapsulation, and shared-secret handling
  remain Planned until the secret-device boundary and deployment profile are
  explicit.
- No OARS API may claim that ordinary Vulkan prevents GPU snapshots.

## 2. What capture and profiling tools can observe

The term “profiler” hides materially different tools:

| Tool behavior | Typical data | Secret risk |
|---|---|---|
| timestamp/counter profiler | durations, occupancy, stalls, cache/bandwidth and performance counters | possible side-channel leakage; not normally a complete buffer dump |
| API capture/replay | API calls, descriptors, shaders, resource contents and revisions needed to replay | may persist keys, entropy, workspace, plaintext, and secret-derived buffers in a capture file |
| source debugger | variables plus local, shared, and global GPU memory at a breakpoint | direct plaintext disclosure |
| GPU/core crash dump | selected execution and memory state needed for post-mortem analysis | capture contents depend on tool/configuration; treat as potentially secret-bearing |
| injected Vulkan layer or modified shader | arbitrary observation or copying within the process authority | complete compromise of secrets available to that process |

NVIDIA documents that its CUDA debugger can view local, shared, and global GPU
memory. Nsight Graphics documents a Resource Viewer memory dump and capture
files that contain API resources for standalone replay. RenderDoc and equivalent
Vulkan capture tools must therefore be treated as hostile to production keys,
even when they are essential engineering tools.

Keeping an intermediate in registers or workgroup memory reduces global-memory
traffic and lifetime. It does not defeat an authorized source debugger, and a
vendor compiler may spill registers to device memory. The emitted SPIR-V cannot
prove the final native compiler's spill behavior on every driver.

## 3. Threat boundary

| Actor/event | Ordinary Vulkan deployment | Required response |
|---|---|---|
| another unprivileged process | relies on OS/driver process isolation | do not export memory or handles; test isolation assumptions on supported platforms |
| same process, plugin, injected layer, debugger, or capture tool | can observe or deliberately disclose accessible plaintext | outside library confidentiality boundary; exclude from production workload and attested image |
| same-user process with debugging/ptrace authority | platform-dependent compromise is plausible | OS hardening and separate service identity; ordinary mode cannot claim resistance |
| host administrator, hypervisor, BMC, or driver owner | not a secrecy boundary on an ordinary system | confidential-compute hardware and attestation are required for this threat |
| basic physical bus or memory observation | ordinary platform-dependent exposure | confidential-compute encrypted/authenticated transport and memory protection where supported |
| sophisticated physical attack | outside the cited NVIDIA CC threat model | no OA claim |
| stale allocation, crash artifact, swap, or host staging | plaintext may outlive the logical operation | bounded host/device erasure, dump policy, no pool reuse before erase completion |
| timing/performance-counter observer | may infer secret-dependent work | constant-work/data-independent design, tooling restrictions, multi-vendor leakage analysis |
| malicious OA caller | can ask the library to reveal its own results or instrument its own process | explicitly out of scope; cryptographic libraries cannot protect secrets from their authorized caller |

The threat model and deployment mode must accompany every security claim. A
statement such as “keys remain on GPU” only identifies placement; it says
nothing about debugger, host-admin, capture, DMA, crash-dump, side-channel, or
post-free visibility.

## 4. NVIDIA CUDA and Confidential Computing

### 4.1 Ordinary CUDA/cuPQC

cuPQC uses device functions so cryptographic work can be fused into surrounding
CUDA kernels and intermediates can remain in registers/shared memory instead of
round-tripping through global memory. This is a valuable exposure-reduction and
performance property. It is not a debugger or snapshot boundary.

cuPQC's security notes report machine-learning-assisted side-channel analysis
and mitigations for known issues. They also report observed execution and
memory-access variations and do not turn those observations into a universal
constant-time or certification claim. OA cannot inherit cuPQC's testing simply
by implementing the same FIPS algorithms.

### 4.2 Full CC-On

NVIDIA documents the following relevant properties for supported Hopper and
Blackwell Confidential Computing configurations:

- code and data in the confidential VM are intended to be protected from the
  host;
- a confidential GPU is passed through exclusively to the confidential VM;
- CPU-to-GPU traffic uses encrypted/authenticated transport, including
  protected bounce-buffer flows on supported Hopper configurations;
- GPU and environment attestation precedes accepting confidential work;
- BMC/management paths are reduced so they cannot access tenant data;
- JTAG is disabled in Confidential Mode;
- hardware performance counters are disabled in full CC-On because they are a
  side-channel risk;
- ephemeral platform encryption keys are inaccessible to the hypervisor and
  destroyed when the VM is torn down or the GPU is reset.

This protects against the external host/hypervisor threat only when the exact
hardware, firmware, driver, VM, mode, attestation claims, transport, and
operational configuration are verified. It is not a property OARS can infer
from the NVIDIA vendor string or from successfully creating a Vulkan device.

### 4.3 CC-DevTools

NVIDIA provides CC-DevTools mode so developers can access performance counters
while encryption paths remain active. It is a development/qualification mode,
not evidence for the full CC-On counter restriction. A capture, trace, or
benchmark produced with DevTools enabled must record that mode and must not use
production secrets.

The production rule is therefore:

1. construct a measured confidential workload without debugger/capture tools;
2. attest the CPU TEE, GPU, firmware, driver, configuration, and DevTools-off
   policy;
3. provision or unwrap application secrets only after successful attestation;
4. reject work and erase any staged material if attestation or policy changes;
5. rotate/destroy the confidential workload and keys at its terminal boundary.

Attestation is freshness- and policy-sensitive. A previously saved successful
report does not prove the state of a later process or GPU session.

## 5. Vulkan protected memory

Vulkan 1.1 protected memory is a distinct mechanism from NVIDIA Confidential
Computing. The Vulkan specification requires protected memory to be
device-visible but not host-visible, and prevents unprotected output from being
a function of protected data. Protected queues may execute compute work, while
queries and indirect execution are restricted according to protected-memory
properties.

This can block ordinary host mapping and some accidental or API-level
declassification. It is not automatically suitable for vkPQC:

- support and usable protected compute queues are capability-dependent;
- protected memory cannot be host-mapped;
- a signature, shared secret, or ciphertext derived from protected input cannot
  simply be written to ordinary unprotected memory under the protected-memory
  access rules;
- query restrictions interfere with ordinary profiling and timing evidence;
- it does not replace confidential-VM and device-attestation guarantees against
  a hostile host, driver stack, or trusted application code;
- a usable output path requires a specific protected consumer or a separately
  specified declassification protocol.

Therefore OARS must not select protected memory merely because an allocation is
secret. It is a separately capability-gated execution mode with an end-to-end
protected consumer. Ordinary host-returning ML-DSA/ML-KEM APIs use the explicit
secret-device contract below and make the correspondingly weaker threat claim.

## 6. OARS secret-device contract

### 6.1 Types and allocation

- Secret keys, shared secrets, entropy, and secret-dependent workspace never
  inhabit public `Matrix`, `Buffer`, public graph constants, or exportable
  external-memory values.
- A private engine-owned secret allocation is non-mappable and non-exportable
  by OARS. This limits ordinary API exposure but is not described as hostile
  debugger resistance.
- Public keys, signatures, and ciphertexts retain semantic public types even
  when their storage is ordinary U8 device memory.
- A workspace inherits secret classification if any byte may depend on a
  secret, including register-spill backing controlled by a driver.
- Secret allocation identifiers, sizes, operation counts, and lifetimes may
  themselves be sensitive metadata and require diagnostic review.

### 6.2 Entropy

- Production entropy originates from a reviewed OS CSPRNG or an admitted
  hardware-backed source. GPU clocks, timestamps, counters, scheduling,
  uninitialized memory, and shader races are never entropy sources.
- Deterministic seeds are accepted only through explicit test/internal
  interfaces used by official vectors and reproducibility tests.
- Entropy is single-use, typed, range-checked, and never cached in a reusable
  semantic graph constant.
- Host staging is locked and erased on every success/failure path to the extent
  described by `SecureBuffer`; this remains a bounded process-memory claim.
- Device entropy is erased only after its final consuming event completes.

### 6.3 Recording, execution, and erasure

- The semantic graph records secret classification, not secret bytes.
- Graph dumps, pipeline-cache keys, labels, structured logs, validation
  messages, errors, panics, and benchmark artifacts never contain key, entropy,
  shared-secret, or secret-workspace contents.
- Command buffers may contain device addresses/descriptor indices and scalar
  lengths. Diagnostics treat those as sensitive metadata and never serialize
  raw native handles.
- Secret inputs remain alive until the exact final consumer event. Erasure is a
  recorded fixed-size overwrite ordered after that consumer with explicit
  compute/transfer stages and access masks.
- Allocation reuse is forbidden until the erasure event completes. Logical
  destruction without a completed overwrite is not reported as erasure.
- Error, device-loss, process-crash, driver-reset, and power-loss paths cannot
  guarantee an application-issued overwrite. Documentation states this limit;
  confidential-compute reset keys may provide a stronger platform boundary
  only when attested.
- Shader workgroup/private memory has no portable explicit zeroization
  guarantee after invocation termination. Minimize lifetime and avoid spills,
  but do not claim complete register/shared/cache erasure.

### 6.4 Capture and debug policy

The runtime may detect known validation layers, capture layers, debug builds,
or environment switches and refuse secret GPU operations unless an explicit
test-key override is active. This is defense in depth only: injected code and
privileged debuggers need not identify themselves.

No production-secret run may enable:

- RenderDoc, Nsight Graphics capture/replay, CUDA source debugging, GPU core
  dumps, GFXReconstruct capture, or equivalent resource capture;
- validation features that dump resource contents;
- process core dumps or crash reporters that collect address-space contents;
- logs or telemetry with raw request, key, signature-randomness, entropy,
  shared-secret, or workspace buffers.

Counter-only profiling still changes the side-channel and timing environment.
It uses public/synthetic keys and records the tool, counter set, permissions,
driver, device, workload, and confidential-compute mode.

## 7. Operational profiles

| Profile | Keys/data | Tooling | Permitted claim |
|---|---|---|---|
| Development | fixed public vectors and disposable test keys only | validation, capture, debugger, shader instrumentation allowed | correctness investigation only |
| Qualification | synthetic/disposable keys only | separate validation, GPU-assisted checks, counter profiling, capture where required | evidence for the exact tested artifact; no production secrecy |
| Ordinary production | production keys allowed by application policy | no debugger/capture/core dump; least privilege and isolated service identity | bounded OARS ownership/erasure; no hostile host-admin confidentiality |
| Confidential production | secrets provisioned only after current attestation | full CC-On/DevTools off, no debugger/capture/core dump | exact attested platform threat model only |

One artifact cannot simultaneously be a capture-debug build and confidential
production evidence. Qualification uses disposable secrets and production
repeats correctness smoke tests without observability tooling.

## 8. Verification requirements

Before admitting a secret-bearing GPU operation:

1. Prove algorithm correctness using official vectors and an independent CPU
   differential oracle.
2. Prove every success and ordinary failure path schedules device erasure and
   prevents pool reuse until completion.
3. Inject allocation, submission, shader, timeout, cancellation, and device-loss
   failures and record which paths cannot guarantee overwrite.
4. Verify logs, graph dumps, errors, panic paths, and benchmark artifacts with
   sentinel secrets and automated secret-pattern scanning.
5. Inspect the emitted SPIR-V and vendor profiler evidence for control-flow,
   memory-access, divergence, and spill hypotheses using disposable keys.
6. Run statistical fixed-vs-random leakage experiments appropriate to each
   supported device/driver. A negative test is bounded evidence, not proof of
   universal constant time.
7. Run Vulkan core, synchronization, and GPU-assisted validation separately
   with test keys.
8. For a confidential-production claim, verify and archive attestation claims,
   CC mode, DevTools state, firmware, driver, CPU TEE, GPU, transport mode,
   workload identity, and provisioning policy.
9. Obtain an independent cryptographic and platform-security review.

## 9. Failure behavior

- Verification rejects malformed and non-canonical inputs and returns false per
  element; infrastructure failure returns `Error` and never masquerades as an
  invalid signature result.
- ML-KEM decapsulation follows FIPS 203 implicit rejection and does not expose a
  ciphertext-validity oracle.
- Entropy failure, unsupported secret storage, capture-policy conflict,
  attestation failure, device loss, or inability to establish required erasure
  semantics fails the operation. There is no hidden CPU fallback.
- A process or device failure that prevents erasure is reported as an
  unconfirmed-erasure security event where reporting remains possible.

## 10. Rejected claims

- “Device-local” means inaccessible to the host or debugger.
- Registers/shared memory cannot be captured or spilled.
- Zeroization eliminates all compiler, register, cache, driver, swap, crash,
  or physical copies.
- cuPQC's side-channel investigation transfers to vkPQC.
- Vulkan protected memory is equivalent to a confidential GPU.
- CC-DevTools evidence proves full CC-On behavior.
- Attestation protects secrets from the attested workload or authorized caller.
- Passing FIPS algorithm vectors constitutes FIPS 140 validation.

## 11. Primary references

- [NVIDIA cuPQC SDK](https://docs.nvidia.com/cuda/cupqc/)
- [NVIDIA cuPQC security notes](https://docs.nvidia.com/cuda/cupqc/additional/security_notes.html)
- [NVIDIA Secure AI with Hopper and Blackwell GPUs](https://docs.nvidia.com/nvidia-secure-ai-with-blackwell-and-hopper-gpus-whitepaper.pdf)
- [NVIDIA Attestation documentation](https://docs.nvidia.com/attestation/)
- [Nsight CUDA debugger state and memory inspection](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html)
- [Nsight Graphics capture overview](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-overview.html)
- [Nsight Graphics Resource Viewer](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-ui.html)
- [Vulkan protected memory](https://docs.vulkan.org/spec/latest/chapters/memory.html#memory-protected-memory)
- [Vulkan protected-memory guide](https://docs.vulkan.org/guide/latest/protected.html)
- [NIST FIPS 203 — ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 — ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
