# OA public documentation sources

**Status:** Current public documentation index

`docs/external` contains material that may ship in source snapshots and feed
the Realm developer site. It must be understandable without the private
architecture, migration, research or device-report tree.

## Contents

- [Documentation](documentation/) — API, example and tutorial authoring and generation contract
- [Tutorials](tutorial/) — public prose paired with checked executable sources
- [Benchmarks](benchmarks/) — reproducible, revision-scoped public measurements
- [Assets](../../sdk/asset/documentation/) — manifested offline documentation and presentation media
- `generated/` — reviewable snapshots consumed by standalone documentation builds
- [Python package readme](pyPIReadme.md) — published `oapython` package description

Executable C++ and Python source remains under `sdk/{cpp,py}`. This tree owns
publication prose and generated reference snapshots; manifested presentation
media and its provenance remain under `sdk/asset/documentation`.

External documents must not reference the private documentation tree, editor
configuration, workstation paths or private release tooling. Validate the
boundary with:

```bash
python3 tools/documentation/checkExternalDocs.py
```
