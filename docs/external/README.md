# OA public documentation sources

**Status:** Current public documentation index

`docs/external` contains material that may ship in source snapshots and feed
the Realm developer site. It must be understandable without the private
architecture, migration, research or device-report tree.

## Contents

- [Documentation](documentation/) — API, example and tutorial authoring and generation contract
- [Tutorials](tutorial/) — public prose paired with checked executable sources
- [Benchmarks](benchmarks/) — reproducible, revision-scoped public measurements
- [Assets](assets/) — approved generated images and presentation media
- `generated/` — reviewable snapshots consumed by standalone documentation builds
- [Python package readme](pyPIReadme.md) — published `oapython` package description

Executable C++ and Python source remains in the repository's top-level
`tutorial/` and `examples/` directories. This tree owns publication prose,
provenance and generated presentation inputs; it does not duplicate the code.

External documents must not reference the private documentation tree, editor
configuration, workstation paths or private release tooling. Validate the
boundary with:

```bash
python3 tools/documentation/check_external_docs.py
```
