# VKL provenance

VKL is OA's collision-safe C fork of
[volk](https://github.com/zeux/volk), kept as a reusable Vulkan loader rather
than an OA runtime API.

- Upstream commit: [`e91ceff7db05f92ac8b498c9631235a6a73566e2`](https://github.com/zeux/volk/commit/e91ceff7db05f92ac8b498c9631235a6a73566e2)
- Upstream Vulkan header revision: `346`
- Upstream snapshot date: 2026-03-13
- Fork date: 2026-03-18
- License: MIT; see `../licenses/volk.txt`
- Upstream snapshot SHA-256: `volk.h`
  `d14d9255cde6149a59314d7ede19e877636cb9b4761384e9812dfa6f7f223edd`,
  `volk.c`
  `b0d038d19ddbfb794bef72ce92353d2146c3dd3695aa717dbc2dda976373b38b`

## OA-local patch manifest

| Patch | Contract |
|---|---|
| Rename exported `volk*` functions to `vkl*`, `Volk*` tables to `Vkl*`, and `VOLK_*` macros to `VKL_*` | Allows OA and a consumer's stock volk to coexist without symbol or macro substitution inside VKL |
| Retain a C ABI and remove upstream optional C++ namespace generation | Keeps VKL usable by C, C++, and Android loader integrations |
| Disable legacy global instance/device dispatch and device prototypes | OA uses immutable per-instance and per-device tables; no engine may replace another engine's dispatch |
| Keep loader-global Vulkan pointers private and expose checked `vklCreateInstance`, instance enumeration, version, and `vklGetInstanceProcAddr` functions | Avoids exporting `vk*` pointer variables that collide with the native Vulkan loader while preserving the complete pre-instance loader surface |
| Load each device table through the owning instance table's `vkGetDeviceProcAddr` | Removes upstream's process-global device-dispatch dependency and permits simultaneous engines |
| Fill core 1.1/1.3 promotion slots from available KHR aliases | Supports extension-backed Android HALs without changing the caller-visible core table |
| Make initialization idempotent, reject a module without `vkGetInstanceProcAddr`, make null custom initialization safe, and null-check table destinations | Fails closed on incomplete or invalid loader setup |

OA-specific loader selection policy is not part of VKL. It lives in
`source/cpp/lib/oa/runtime/loader.cpp`, where process-global selection is
serialized and an incompatible second loader is rejected.

## Update procedure

1. Fetch an exact upstream commit and record its full SHA and pristine source
   hashes before applying patches.
2. Regenerate or reapply every patch in the manifest; do not hand-copy new
   Vulkan commands.
3. Verify `VKL_HEADER_VERSION` against upstream `VOLK_HEADER_VERSION`.
4. Compile the header as C11 and C++20 with `VK_NO_PROTOTYPES` and all supported
   platform defines.
5. Run loader failure/idempotence tests and the two-engine dispatch-isolation
   gate, including distinct instance/device tables and Android custom-loader
   selection.
6. Run Vulkan core and synchronization validation on the product runtime.
7. Record the post-patch hashes below.

## Post-patch checksums

| File | SHA-256 |
|---|---|
| `vkl.h` | `d57760e6426e54fbb06da7f7270e6680dfd645cd4418c9d5d9a90e9678f35873` |
| `vkl.c` | `13b9e2aabfa67d91351c515858a534af88e66da85fd8a1cff97a9268f3bad02b` |
| `../licenses/volk.txt` | `04a0693a84f19e53d281ca98bbb0c86ca77251ab13769c6168e6684feb9a1436` |
