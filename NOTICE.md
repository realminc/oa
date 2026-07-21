# NOTICE — Third-Party Software and Attributions

OA is licensed under the Business Source License 1.1 (see `LICENSE`). This file
records software incorporated into OA distributions, optional build/runtime
dependencies, and the provenance of interoperability data and formats. Each
third-party work remains under its own license. Listing a project or mark does
not imply endorsement by, or affiliation with, its owners.

Release packages install this file, OA's `LICENSE`, and the package-manager
copyright files for direct dependencies compiled into the release. Source-tree
copies retain their upstream notices at the paths listed below.

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
| **libadrenotools** | `Apps/Android/OaMobileLab/third_party/libadrenotools/*` (Android only) | Billy Laws ([bylaws/libadrenotools](https://github.com/bylaws/libadrenotools)) | BSD-2-Clause |

The volk and VMA copies are **modified** (fork/rename/split for OA's build); their MIT
copyright and permission notices are preserved in `OaVk.h` and `OaVma.h` respectively,
and each split VMA module carries an SPDX attribution line pointing back to that notice.
GLM's complete license choices are retained at `Source/ThirdParty/glm/copying.txt`;
libadrenotools and its linkernsbypass copy retain their `LICENSE` files in place.

## Direct build and link dependencies

These dependencies are resolved by vcpkg or the host package manager; they are
not copied into OA's source tree. Whether a component is present in a particular
binary depends on that build's feature flags and platform. Copyright files from
the active vcpkg installation are copied into release packages automatically.

| Component | OA use | License |
|---|---|---|
| yaml-cpp | YAML configuration and model metadata | MIT |
| CLI11 | command-line interfaces | BSD-3-Clause |
| {fmt} | formatting | MIT |
| Highway | host SIMD implementation | Apache-2.0 |
| utf8proc | Unicode text processing | MIT |
| Vulkan-Headers | Vulkan API declarations | Apache-2.0 OR MIT |
| SDL3 | windows, input and camera integration | Zlib; selected configurations also contain MIT/Apache-2.0 code |
| liboqs | optional post-quantum cryptography | MIT, with separately licensed code identified by liboqs |
| DuckDB | optional embedded database | MIT |
| nanobind | optional Python extension bindings | BSD-3-Clause |

Slang (Apache-2.0 with LLVM exception) compiles OA shaders during the build but
is not linked into the OA runtime. GoogleTest (BSD-3-Clause) is used only by the
test targets. NVTX headers (Apache-2.0 with LLVM exception) are used only for
optional profiling markers when an NVTX installation is detected.

On Linux, OA may dynamically link host-provided Vulkan loader, liburing, SDL3,
PipeWire and libportal libraries. OA packages do not redistribute those system
libraries; their licenses and notices are supplied by the operating-system
packages. `ldd`/the platform package metadata is the authority for the exact
dynamic dependency closure of a given binary.

## Reference implementations and test oracles

PyTorch, TensorFlow, OpenCV and FFmpeg are used in documentation as behavioral,
API, or differential-testing references. OA does not bundle or link their source
or libraries. Their licenses therefore are not licenses of incorporated OA
components and are intentionally not reproduced as if they were dependencies.

In particular, some video tests invoke a separately installed `ffmpeg` executable
to build fixtures or compare decoded frames. FFmpeg is not invoked by the OA
runtime and no `libav*` library is linked. FFmpeg is LGPL-2.1-or-later by default;
an FFmpeg build that enables its optional GPL components is covered by GPL-2.0-or-later.

## Regenerating the derived data (maintainers)

The baked tables in `Extensions/Private/{Rig,Retarget}/*.inc` are checked in; building
OA does **not** require the generators. `Tools/Gen3dAnim/*.py` can rebuild them from a
local, privately-held copy of the source definitions (pointed at via the `OA3D_*`
environment variables). Those private sources are not distributed with OA.
