# OA release notes

**Status:** Current public release index

Release notes describe one immutable public source and package checkpoint. They
record the complete shipped scope, compatibility impact, verification evidence,
known limitations, and upgrade guidance for that version. The
[changelog](../../../CHANGELOG.md) remains the concise chronological list of
notable deltas; the root [README](../../../README.md) remains an evergreen product
introduction.

## Releases

- [v0.7.20](v0.7.20.md) — current public development preview
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
