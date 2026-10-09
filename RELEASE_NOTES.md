# OA v0.8.6 — Rust development preview

## Summary

v0.8.6 completes Rust registry packaging and repairs the hosted SDL3 linker
failure after v0.8.4 and v0.8.5 failed before package publication. Native and
Python builds use the checksum-verified Slang 2026.14 compiler and bundled SDL3.

OA provides one Rust-owned Vulkan Engine, typed values, semantic and executable
graphs, and Rust/Python domain APIs. All GPU capabilities remain Experimental.

The preview includes Matrix operations and autograd, schema-generated ML
functions and layers, training/checkpoint workflows, image and audio operations,
Vision metrics, media sessions, Plot and Viewer, and cryptography/PQC operations.
Capability availability depends on device admission and each operation's checks;
source availability does not imply physical-device or security qualification.

## Compatibility and migration

- Rust consumers depend on `oarust`; its library namespace is `oa`. The
  historical `oarust` 0.0.1 version remains a name reservation.
- Install the Python distribution with `pip install oapython==0.8.6`; import `oa`.
- The Rust API replaces the C++ runtime and is not a binary-compatible upgrade.
- Strict execution uses Vulkan 1.3. Bounded Compatibility compute admits Vulkan
  1.2 devices that satisfy the runtime's actual feature and limit requirements.
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
separate gates. Rust source consumers need the documented compiler and native build prerequisites.

## Upgrade

Use the v0.8.6 source and matching SDK/wheel artifacts together. Rebuild Rust
consumers and native Python extensions rather than reusing C++ or earlier
experimental binaries. Verify downloads with `SHA256SUMS.txt`.
