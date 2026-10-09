# Changelog

## 0.8.5

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
