# OA release notes

**Status:** Current public release index

Release notes describe one immutable public source and package checkpoint. They
record the complete shipped scope, compatibility impact, verification evidence,
known limitations, and upgrade guidance for that version. The
[changelog](../../../CHANGELOG.md) remains the concise chronological list of
notable deltas; the root [README](../../../README.md) remains an evergreen product
introduction.

## Releases

- [v0.7.28](v0.7.28.md) — current public development preview; Vulkan Linear Math foundation
- [v0.7.27](v0.7.27.md) — blocked hosted candidate; generator namespace policy mismatch
- [v0.7.26](v0.7.26.md) — Python compute-fallback repair
- [v0.7.25](v0.7.25.md) — C++ release snapshot; Python wheel publication blocked by a fallback-warning format mismatch
- [v0.7.24](v0.7.24.md) — native-video source snapshot; hosted publication blocked by shader authority and validation logging
- [v0.7.23](v0.7.23.md) — foundation public development preview
- [v0.7.22](v0.7.22.md) — source snapshot; hosted publication blocked by the ASAN test timeout
- [v0.7.21](v0.7.21.md) — foundation source snapshot; hosted publication blocked by the dependency ratchet
- [v0.7.20](v0.7.20.md) — hosted sanitizer-target repair
- [v0.7.19](v0.7.19.md) — foundation source snapshot; hosted publication failed
- [v0.7.18](v0.7.18.md) — paired Transformer SDK example

OA follows semantic versioning for public releases:

- **major** — incompatible stable public contract or artifact-format change;
- **minor** — backwards-compatible capability added to a stable line;
- **patch** — backwards-compatible correction, performance improvement,
  documentation, or packaging change.

Before 1.0, a minor release may deliberately change source, ABI, or artifact
contracts. Such changes must be named under **Compatibility and migration**;
"pre-1.0" is not permission to hide them. A patch release must not silently
break a documented public contract.

The tag `vX.Y.Z`, root `VERSION`, package metadata, generated language version,
release title, and artifact names identify the same public version. Published
tags and assets are immutable.
