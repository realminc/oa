# OA data tools

**Status:** Maintained host tooling

`checkAssets.py` verifies the complete checked-in `sdk/asset` inventory.
`manage.py` lists, fetches, verifies, and locates optional dataset packs declared
by `sdk/data/packs.toml`.

Data acquisition is intentionally host-side Python using only the standard
library. OA runtime initialization and installed C++/Python APIs never open the
network. The native `datasetctl` application has a separate responsibility: it
packs and inspects OA `.oad` archives after input data already exists locally.

```bash
python3 tools/data/checkAssets.py
python3 tools/data/manage.py list
python3 tools/data/manage.py fetch fashionMnist
python3 tools/data/manage.py verify fashionMnist
python3 tools/data/manage.py path fashionMnist
```

Set `OA_DATA_DIR` or pass `--data-root` to choose the storage root. Fetches use
temporary files below the destination pack, verify compressed and expanded
size/SHA-256 pins, and publish by atomic rename.
