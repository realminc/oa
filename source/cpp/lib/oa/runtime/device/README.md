# OA runtime device implementation

Status: **Implementation map; not an architecture authority**
Last source audit: 2026-07-23

The root project README and installed public headers define the supported
contract. This file only maps that contract onto the implementation directory.

This directory implements the low-level Vulkan device used and owned by
`oa::Engine`. Applications should create an engine rather than building a parallel
device/runtime owner.

## Files

| File | Responsibility |
|---|---|
| `device.cpp` | instance-backed physical-device selection, compatibility `oavk::Device` creation and destruction |
| `deviceBuilder.cpp/.h` | feature-module orchestration, queue planning, logical-device creation and capability publication |
| `deviceUtils.cpp` | device survey, properties, rating and capability helpers |
| `computeDevice.cpp` | compute-device capability view |
| `renderDevice.cpp` | graphics/presentation capability view |
| `featureModule.h` | internal module interface |
| `features/coreFeatures.cpp` | required modern-compute feature chain and external-memory probes |
| `features/mlFeatures.cpp` | BF16, integer-dot, cooperative and DGC capabilities |
| `features/visionFeatures.cpp` | Vulkan Video codec/queue and YCbCr capabilities |
| `features/renderFeatures.cpp` | swapchain and presentation-related capabilities |
| `features/audioFeatures.cpp` | currently contributes no device-only audio feature |

Core, ML, Vision and Render modules are implemented capability contributors;
they are not future PR stubs. Their presence in the feature chain does not prove
that every higher-level operation, codec, vendor, or platform is supported.

## Invariants

- One builder creates one logical device for the owning engine.
- Features are queried before they are enabled.
- Optional extensions enter the create list only with their required feature
  bits and dependencies.
- Compute, transfer, graphics/present and video queues are planned explicitly.
- Headless graphics and WSI presentation remain different modes.
- Published capability flags describe enabled state, not extension-name
  advertisement alone.
- Unsupported required capability or `vkCreateDevice` failure returns an
  `oa::Status`; there is no silent device or CPU fallback.

Tests and live validation evidence belong in device-specific qualification and
release reports, not in this directory README.
