# Vulkan Video profile fixtures

These small, profile-explicit clips are the checked-in fixtures used by OA's
Vulkan Video decoder and container tests. They are derived from the Shibuya
Crossing benchmark source and cover only the four profiles listed in
`manifest.tsv`.

## Provenance and license

- Original: [Shibuya Crossing, Tokyo, Japan](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm)
- Author: Basile Morin
- License: [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Derivation: first 60 frames at 1280×720 and 30 FPS, without audio

The sibling dataset workspace owns the regeneration recipe under
`video/profiles/generate.sh`. Encoded bytes can change with FFmpeg and codec
library versions, so replacement fixtures require an independent decode
oracle and an intentional digest update in `manifest.tsv`.

These files prove only their encoded profile cases. They are not evidence for
other bit depths, chroma modes, interlaced layouts, film grain, or device
support.
