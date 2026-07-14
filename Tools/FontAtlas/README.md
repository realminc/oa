# Font Atlas Generator

`generate_sdf_atlas.py` deterministically converts the vendored IBM Plex Sans
Regular font into the embedded single-channel SDF atlas used by `OaTextAtlas`.
Pillow is required only when regenerating the committed include, not when
building or running OA.

```sh
python3 Tools/FontAtlas/generate_sdf_atlas.py \
  --font Asset/Font/IBMPlexSans/IBMPlexSans-Regular.ttf \
  --output Source/Private/Oa/Ui/Generated/IBMPlexSansSdf.inc
```
