# OA v0.10.0 — Rust development preview

## Summary

v0.10.0 adds submit-time graph fusion over the joined executable graph. The
public recording API is unchanged; a capture that records a chain of node
operations now submits fewer, structurally fused dispatches whose kernels
materialize every buffer the unfused sequence wrote, so autograd saves and
external readers observe identical storage.

Two passes run before submission. An elementwise tape pass fuses adjacent
legal F32 chains into one `MatrixFusedChainF32` dispatch, gated at 32768
elements on measured routing evidence. An epilogue pass fuses structural
patterns into lowering-only kernels: `linear → add` residual, `linear →`
one of nine unary activations, `add → layer_norm`, a z=3 shared-input
`linear` triple for QKV projections, `embedding + embedding + add` with
mixed U8/U32 indices, and `sum_axis → scale`. Record-site collapses remove
purely internal dispatches in the cross-entropy, pointwise-loss, and PPO
`sum → scale` tails and in the dual reduction behind norm backward.
`fused_op_count` and `fused_node_count` diagnostics report the rewritten
graph.

Measured on the 16-model NLP tutorial suite on Intel HSW GT2: transformer
graphs drop from 67 to 56 nodes and MoE graphs from 82 to 73, with
unchanged convergence and a measured GPU-side improvement on the fused
residual path. Flash causal attention qualified roughly 4.8× slower than
the standard provider at the tutorial attention shape on this device, so
`AttentionBackend::Auto` correctly remains on the standard provider.

v0.9.1 consolidated runtime kernel identity and BLAS routing under
`runtime/dna` and added the measured `mat_mul_nt_gemv` decode candidate.
v0.9.0 completed OA's migration to its own `vkrust` Vulkan bindings.

OA provides one Rust-owned Vulkan Engine, typed values, semantic and
executable graphs, and Rust/Python domain APIs. All GPU capabilities
remain Experimental. Capability availability depends on device admission
and each operation's checks; source availability does not imply
physical-device or security qualification.

## Compatibility and migration

- Rust consumers depend on `oarust`; its library namespace is `oa`. The
  historical `oarust` 0.0.1 version remains a name reservation.
- Install the Python distribution with `pip install oapython==0.10.0`;
  import `oa`.
- The Rust API replaces the C++ runtime and is not a binary-compatible
  upgrade.
- Fusion changes the physical dispatch graph, not recorded semantics.
  `OA_DISABLE_EPILOGUE_FUSION` and `OA_DISABLE_ELEMENTWISE_FUSION` restore
  the unfused layout for diagnosis.
- Strict execution uses Vulkan 1.3. Bounded Compatibility compute admits
  Vulkan 1.2 devices that satisfy the runtime's actual feature and limit
  requirements.
- Heavyweight cryptography and device tests are opt-in via
  `OA_TEST_SLOW=1`.
- Internal engineering documents and private release/editor tooling are
  excluded from the public source tree and source archives.

## Verification

Source validation covers generation/drift, formatting, clippy, Rust tests,
Python tests and explicit public-snapshot sanitization. Fused kernels carry
hardware tests proving numerical identity against the unfused capture on
named devices. Tagged CI independently builds this exact source, runs
software-Vulkan smoke checks, installs the wheel in a clean environment,
and verifies the published artifact checksums. Hardware-dependent tests
remain distinct from software-Vulkan validation.

## Known limitations

The API remains unstable. Compatibility presentation is not yet qualified;
Vulkan Video requires suitable device/driver capabilities. Physical GPU
PQC qualification, independent security review, and competitive
performance remain separate gates. Fusion currently admits the measured
elementwise and epilogue patterns only; broadcast operands, backward
elementwise chains, and additional producers remain unfused. The pipeline
cache is best-effort: unreadable or driver-rejected blobs are discarded
and rebuilt. Rust source consumers need the documented compiler and
native build prerequisites.

## Upgrade

Use the v0.10.0 source and matching SDK/wheel artifacts together. Rebuild
Rust consumers and native Python extensions rather than reusing earlier
artifacts. Verify downloads with `SHA256SUMS.txt`.
