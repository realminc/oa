# Font Atlas Generator

`generateAtlas.py` deterministically converts the vendored IBM Plex Sans
Regular/SemiBold and Intel One Mono Regular faces into the embedded
single-channel hinted coverage atlas used by the text renderer.

The atlas contains exact UI pixel-size strikes from 9 through 32 pixels. The
generator reads each face's Unicode `cmap`, embeds only the bounded editor/UI
coverage present in at least one face, and records support separately for each
font. Missing glyph boxes are never accepted as font coverage. IBM Plex Sans
Regular supplies the fallback for SemiBold, both Sans faces supply Greek and
Cyrillic fallback for Intel One Mono, and Intel One Mono supplies its
box-drawing and block characters.

Pillow uses FreeType to rasterize those build-time inputs. OA does not load or
rasterize fonts at runtime; the generated include embeds the exact three source
faces used by HarfBuzz for positioning and OpenType substitution. HarfBuzz
shapes only the supported left-to-right runs; the generated grayscale atlas
remains OA's sole glyph raster source. The standard vcpkg HarfBuzz package may
also install its FreeType interop dependency, but OA does not call that API.

```sh
python3 tools/gen/font/generateAtlas.py \
  --sans-font sdk/asset/font/IBMPlexSans/IBMPlexSans-Regular.ttf \
  --sans-semibold-font sdk/asset/font/IBMPlexSans/IBMPlexSans-SemiBold.ttf \
  --mono-font sdk/asset/font/IntelOneMono/IntelOneMono-Regular.ttf \
  --output source/cpp/lib/oa/ui/generated/textCoverageAtlas.inc
```

Regeneration with unchanged font bytes, Pillow version, and arguments must
produce an empty diff. The current evidence baseline is Pillow 11.3.0 with
FreeType 2.13.3; the generated header records those versions and all three
source-font hashes.
