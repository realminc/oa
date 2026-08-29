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

## Local real-time video fixture

`videoRealtime` is an opt-in, non-commercial qualification pack sourced from
the Ultra Video Group ReadySetGo sequence. The downloaded master contains 600
native progressive 3840x2160 frames at 120 fps. OA does not redistribute it.

```bash
python3 tools/data/manage.py fetch videoRealtime
python3 tools/data/deriveVideoRealtime.py
```

The derivation produces eight exact 60/1 fps variants from the even-numbered
source frames. Content-addressed copies of the derivatives are checked under
`sdk/asset/video/clip/`; the data pack retains the non-redistributed master and
supports deterministic regeneration:

- `1080p60-h264` encodes H.264 High 4.2;
- `1080p60-h265` encodes H.265 Main 4.1;
- `1080p60-av1` encodes low-delay AV1 Main 4.1;
- `1080p60-vp9` encodes low-delay VP9 Profile 0 level 4.0;
- `2160p60-av1` preserves native pixels and encodes AV1 Main 5.1;
- `2160p60-h264` preserves native pixels and encodes H.264 High 5.2;
- `2160p60-h265` preserves native pixels and encodes H.265 Main 5.1;
- `2160p60-vp9` preserves native pixels and encodes VP9 Profile 0 level 5.0.

The full UHD corpus is checked independently of one machine's capabilities.
The current Intel TGL driver reports H.264 decode level 5.1, so it rejects the
exact H.264 High 5.2 row. The fixture never bypasses that device level contract.
Select one variant when iterating:

```bash
python3 tools/data/deriveVideoRealtime.py --variant 1080p60-av1
python3 tools/data/deriveVideoRealtime.py --variant 2160p60-h265 --verify-only
```

No variant interpolates or duplicates frames. The tool fails unless the
source/output size and SHA-256, codec/profile/level, stream contract, decoded
frame count, 300/300 unique-frame cadence, and an independent source-to-output
PSNR floor all pass.

The accepted output checksums are tied to the exact FFmpeg and encoder versions
used by the derivation tool. Different encoder versions must be admitted as a
deliberate fixture revision, not silently relabeled with an accepted checksum.
The source and derivatives remain subject to
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
