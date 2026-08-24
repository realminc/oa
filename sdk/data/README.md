# Optional dataset packs

**Status:** Maintained descriptor authority

`packs.toml` is the source-kit authority for optional external datasets. It
contains provenance, license, immutable source revision, download and expanded
sizes, SHA-256 hashes, and the exact path presented to SDK consumers.

The manifest contains no dataset payload. Use `tools/data/manage.py` explicitly;
library code, tutorials, tests, and CMake never fetch data as a side effect.
Large private/research corpora remain in independently versioned storage and do
not enter this manifest unless a stable redistributable pack is intentionally
published.
