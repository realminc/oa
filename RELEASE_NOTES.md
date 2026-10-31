# OA v0.9.0 — Rust development preview

## Summary

v0.9.0 completes OA's migration to its own `vkrust` Vulkan bindings. The Ash
and vk-mem dependencies are removed; device ownership, allocator
synchronization, capability admission, and teardown run on OA's native binding
layer. The public `oa` namespace is unchanged.

The Vulkan pipeline cache now persists under `var/vk` and admitted shader
pipelines can preload in parallel across compute, graphics, and UI creation
(`OA_SHADER_PRELOAD`, `OA_SHADER_LOAD_THREADS`). Python NLP tutorials run the
same native `ItTraining` capture/replay lifecycle as the Rust tutorials.
Signature-record generation now covers methods, Python bindings, and
constructor vocabulary across handwritten, template-embedded, session, codec,
and export surfaces.

OA provides one Rust-owned Vulkan Engine, typed values, semantic and executable
graphs, and Rust/Python domain APIs. All GPU capabilities remain Experimental.
Capability availability depends on device admission and each operation's
checks; source availability does not imply physical-device or security
qualification.

## Compatibility and migration

- Rust consumers depend on `oarust`; its library namespace is `oa`. The
  historical `oarust` 0.0.1 version remains a name reservation.
- Install the Python distribution with `pip install oapython==0.9.0`; import `oa`.
- The Rust API replaces the C++ runtime and is not a binary-compatible upgrade.
- The Vulkan implementation no longer uses Ash or vk-mem; this is an internal
  dependency change with no public API surface impact.
- Strict execution uses Vulkan 1.3. Bounded Compatibility compute admits Vulkan
  1.2 devices that satisfy the runtime's actual feature and limit requirements.
- Heavyweight cryptography and device tests are opt-in via `OA_TEST_SLOW=1`.
- Internal engineering documents and private release/editor tooling are excluded
  from the public source tree and source archives.

## Verification

Source validation covers generation/drift, formatting, clippy, Rust tests,
Python tests and explicit public-snapshot sanitization. Tagged CI independently
builds this exact source, runs software-Vulkan smoke checks, installs the wheel
in a clean environment, and verifies the published artifact checksums.
Hardware-dependent tests remain distinct from software-Vulkan validation.

## Known limitations

The API remains unstable. Compatibility presentation is not yet qualified;
Vulkan Video requires suitable device/driver capabilities. Physical GPU PQC
qualification, independent security review, and competitive performance remain
separate gates. The pipeline cache is best-effort: unreadable or
driver-rejected blobs are discarded and rebuilt. Rust source consumers need the
documented compiler and native build prerequisites.

## Upgrade

Use the v0.9.0 source and matching SDK/wheel artifacts together. Rebuild Rust
consumers and native Python extensions rather than reusing earlier artifacts.
Verify downloads with `SHA256SUMS.txt`.
