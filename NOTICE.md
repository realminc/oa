# NOTICE — Third-Party Trademarks & Attributions

OA is licensed under the Business Source License 1.1 (see `LICENSE`). This file
records third-party trademarks and the provenance of interoperability data/formats
that OA supports. Listing a mark here is attribution and nominative (compatibility)
use only — it does **not** imply endorsement by, or affiliation with, the owners.

## Epic Games — MetaHuman, CitySample, Unreal Engine, "Manny" / "Quinn"

The skeletal rig and reference-pose data under `Extensions/{Rig,Retarget,Anim}`
(e.g. the 64-joint clean base skeleton and the UE-mannequin t-pose/a-pose tables)
are derived from Epic Games' **MetaHuman** and **CitySample** sample content and the
Unreal Engine mannequins **"Manny"** and **"Quinn"**.

- MetaHuman, CitySample, Unreal Engine, Manny, and Quinn are trademarks or
  registered trademarks of **Epic Games, Inc.**
- MetaHuman characters and animations are, since the 2025 licensing update
  (Unreal Engine 5.6), classified as "non-engine products" that may be used outside
  Unreal Engine. Use of MetaHuman-derived data is subject to the MetaHuman license
  (<https://www.metahuman.com/license>) and the Unreal Engine EULA. OA's rig/anim
  tooling is an **interoperability bridge** for content authored in these ecosystems;
  it neither includes nor redistributes Unreal Engine or the MetaHuman Creator.

## Autodesk — HumanIK, Maya, MotionBuilder, FBX

- **HumanIK**, **Maya**, **MotionBuilder**, and **FBX** are trademarks or registered
  trademarks of **Autodesk, Inc.**
- OA implements a HumanIK-**compatible** characterization (slot names + ids) used as a
  retarget interop hub. This is a naming/mapping convention only — OA does **not**
  include, link, or redistribute the Autodesk HumanIK SDK / middleware, which Autodesk
  licenses separately.
- OA's FBX export is a clean-room ASCII writer for interoperability. It does **not**
  use or include the Autodesk FBX SDK.

## Other formats

- **USD** (Universal Scene Description) support is written against the open USD format;
  Pixar/OpenUSD are not affiliated with OA.

## Vendored open-source components

OA bundles the following third-party libraries. Each is used under its own permissive
license; the full copyright/permission text is retained in the corresponding source
files. OA is grateful to their authors.

| Component | In OA as | Author | License |
|---|---|---|---|
| **volk** (Vulkan meta-loader) | `Source/…/Runtime/OaVk.{c,h}` (hard fork; `volk*` → `OaVk*`) | Arseny Kapoulkine ([zeux/volk](https://github.com/zeux/volk)) | MIT |
| **Vulkan Memory Allocator (VMA)** | `Source/…/Runtime/Vma/*` + `OaVma.{h,cpp}` (vendored; umbrella header split into modules, `vma*` → `OaVma*`) | Advanced Micro Devices, Inc. ([GPUOpen VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)) | MIT |
| **GLM** (OpenGL Mathematics) | `Source/ThirdParty/glm/*` | G-Truc Creation ([g-truc/glm](https://github.com/g-truc/glm)) | MIT (The Happy Bunny / MIT) |
| **miniaudio** | `Source/ThirdParty/miniaudio/*` | David Reid ([mackron](https://github.com/mackron/miniaudio)) | Public domain or MIT-0 |
| **stb** (`stb_image`, …) | `Source/ThirdParty/stb/*` | Sean Barrett ([nothings/stb](https://github.com/nothings/stb)) | Public domain or MIT |

The volk and VMA copies are **modified** (fork/rename/split for OA's build); their MIT
copyright and permission notices are preserved in `OaVk.h` and `OaVma.h` respectively,
and each split VMA module carries an SPDX attribution line pointing back to that notice.

## Regenerating the derived data (maintainers)

The baked tables in `Extensions/Private/{Rig,Retarget}/*.inc` are checked in; building
OA does **not** require the generators. `Tools/Gen3dAnim/*.py` can rebuild them from a
local, privately-held copy of the source definitions (pointed at via the `OA3D_*`
environment variables). Those private sources are not distributed with OA.
