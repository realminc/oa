# Changelog

## 0.9.0

- Migrate the Vulkan backend from Ash and vk-mem to native `vkrust` bindings,
  preserving device ownership, allocator synchronization, capability admission,
  and the public `oa` namespace.
- Persist the Vulkan pipeline cache under `var/vk` and preload admitted shader
  pipelines in parallel (`OA_SHADER_PRELOAD`, `OA_SHADER_LOAD_THREADS`) across
  compute, graphics, and UI pipeline creation.
- Generate methods, Python bindings, and constructor vocabulary from signature
  records, covering handwritten, template-embedded, session, codec, and export
  surfaces.
- Restore native `ItTraining` capture/replay for Python NLP tutorials with
  progress, GPU timing, metrics logging, and checkpoint roundtrips matching the
  Rust tutorials.
- Add a parameterized GPU trading research environment with PPO collection and
  minibatch replay, bounded linear attention, and execution and observation
  corrections.
- Gate heavyweight cryptography and device tests behind `OA_TEST_SLOW=1`.
- Extend cryptography and columnar dataset support; repair consumer builds,
  first-use pipelines, and viewer composition.
- Strip private changelog sections from public snapshots and normalize
  workspace formatting.

See [release notes](RELEASE_NOTES.md) for compatibility and remaining limitations.

## 0.8.6

- Publish the real `oarust` crate with an audited archive and unchanged `oa` namespace.
- Bundle SDL3 and gate registry publication on native and portable-wheel validation.
- Pin and checksum Slang 2026.14 for native and Python CI builds; repair the
  compiler mismatch that prevented v0.8.4 package publication.
- Consolidate the public Rust implementation into its first release checkpoint.
- Provide schema-owned Matrix, ML, autograd, Image, Audio and Vision APIs.
- Include runtime compatibility profiles, explicit device selection and
  software-Vulkan validation paths, media and cryptography/PQC integration.
- Correct Python NLP smoke controls and fresh-model checkpoint restoration.
- Exclude internal documents and private engineering tooling from publication;
  enforce public-source inventory checks in CI.
- Remove obsolete generator previews and the local crates.io reservation scaffold.

See [release notes](RELEASE_NOTES.md) for compatibility and remaining limitations.
