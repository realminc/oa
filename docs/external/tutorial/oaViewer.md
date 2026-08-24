# Image viewing with `oa::Viewer`

**Status:** Current source-tree tutorial

**Updated:** 2026-07-23

`oa::Viewer` is the canonical application. It owns one window/presenter session
and either creates one engine or borrows the caller's engine.

## C++ file example

Source: `sdk/cpp/tutorials/vision/tutorialViewerImage.cpp`

```cpp
#include <oa/ui/viewer.h>

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "asset/image/SpaceCathedral.jpg";
  oa::Viewer viewer(path);
  return viewer.run().isOk() ? 0 : 1;
}
```

Build and run:

```bash
ninja -C build/release TutorialViewerImage
bin/release/tutorial/vision/tutorialViewerImage asset/image/SpaceCathedral.jpg
```

The Viewer owns its UI, input and compose resources and borrows presentation
through `oa::Presenter`. Closing the Viewer releases presentation resources
before the engine closes.

## Python resource example

Source: `sdk/py/tutorials/vision/tutorialVisionViewer.py`

```python
from oa import *

asset = Paths.asset("image/SpaceCathedral.jpg")
image = FnImage.decodeFile(asset)
Viewer.show(image, title="Space Cathedral · original")

resized = FnImage.resize(image, 836, 471)
Viewer.show(resized, title="Space Cathedral · 836×471")
```

Install the source package and run the checked tutorial:

```bash
python -m pip install .
python sdk/py/tutorials/vision/tutorialVisionViewer.py
```

The Python host owns one lazy process-scoped engine. Each blocking `Show` call
borrows that engine and creates a fresh window/surface/presenter session; it
does not recreate the Vulkan device. Close the original window to continue to
the resized image. Automated smoke runs set `OA_UI_MAX_FRAMES=1`.

## Headless example

Source: `sdk/cpp/tutorials/vision/tutorialViewerImageHeadless.cpp`

The headless path creates a compute-only engine, loads an `oa::Texture`, records
texture operations into the selected context, submits explicitly and saves the
result through `oa::FnImage::saveTextureFile`.

```bash
ninja -C build/release TutorialViewerImageHeadless
bin/release/tutorial/vision/tutorialViewerImageHeadless \
  asset/image/SpaceCathedral.jpg /tmp/oa-image.png
```

Headless means no window or swapchain. It does not imply a CPU fallback or
software Vulkan implementation.

## Controls

The current navigation help is printed by the application. Common controls are
fit, 100% zoom, zoom in/out, pan, channel selection and `Q`/Escape to quit.
The exact binding table belongs to the live Viewer/navigation configuration,
not a copied tutorial list.

## Related API

- [`oa::Viewer` public header](../../../source/cpp/include/oa/ui/viewer.h)
- [`oa::Image` public header](../../../source/cpp/include/oa/ui/image.h)
- [Vision public umbrella](../../../source/cpp/include/oa/vision.h)
- [Realm developer API](https://dev.realm.software/vision/reference/api)
