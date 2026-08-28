# Vulkan Memory Allocator provenance

OA vendors a derived C++ snapshot of AMD Vulkan Memory Allocator (VMA).

- Upstream: <https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator>
- Upstream commit: `41a3630a3f8b3bbc0117596945a272cbc70430fe`
- Upstream date: 2026-02-23
- Upstream `include/vk_mem_alloc.h` SHA-256:
  `f71e856c89a41158c47b70f267578856a2d5bb4ffa3ff4a51f4e5f59c6fe91a3`
- Upstream development version: 3.4.0
- License: MIT; centralized at `../licenses/vma.txt`

## OA-local patch manifest

The fork remains mechanically traceable to upstream allocator algorithms. OA's
changes define integration, dependency, naming, and validation boundaries:

| Patch | Reason |
|---|---|
| Split the implementation into `detail/` headers included by one `vma.cpp` | Keep a single implementation translation unit without exposing the monolithic implementation header |
| Disable static and dynamic Vulkan symbol discovery | `oa::Engine` supplies its per-device dispatch table explicitly; VMA does not own loader state |
| Keep upstream-derived `Vma*` storage names and `VMA_*` configuration names | Preserve update traceability and avoid a second mechanical prefix dialect |
| Give derived entry points C++ linkage in private `vma::detail` | Prevent exported C-symbol collisions when an application also links pristine VMA |
| Add the `vma.hpp` camel-case C++ façade | OA consumers use `vma::Allocator` and `vma::createBuffer`, with always-on pointer and value contracts |
| Replace compiled hosted-library synchronization, atomic, algorithm, forwarding, and swap dependencies with OA foundation primitives | Keep the vendored implementation inside OA's foundation dependency boundary |
| Route default host allocation, byte copying, filling, and C-string operations through OA foundation primitives | Reuse the qualified OA allocation and memory policy while preserving application-provided Vulkan allocation callbacks |
| Fail closed on invalid host alignment, allocation-size multiplication, allocation failure, and container-growth overflow | Prevent release-build null placement, wrapped allocation sizes, and wrapped growth capacities |
| Value-initialize Vulkan/function-pointer aggregates and fill pointer arrays by value | Avoid relying on an all-bits-zero representation for C++ pointers |
| Use compiler bit-count intrinsics with portable fallbacks | Remove the remaining hosted `<bit>` dependency without changing bit-operation semantics |

Placement `new` remains C++ language support. Upstream opaque-handle `_T` names
and internal class names are intentionally retained; restyling allocator
algorithms would destroy useful correspondence without improving the public
surface.

## Update procedure

1. Fetch and verify an exact upstream commit, never a moving branch or version
   string alone.
2. Verify the pristine `vk_mem_alloc.h` SHA-256 before transforming it.
3. Re-split the implementation and reapply the patch manifest above.
4. Confirm an unchanged refresh produces no unexplained algorithm diff.
5. Build OA and an application that links pristine VMA beside OA; inspect the
   final symbols for collision isolation.
6. Run `TestVma`, `TestVmaInternals`, `TestAllocator`, ASAN/UBSAN, Vulkan core
   validation, and synchronization validation.
7. Compile `benchVma.cpp` against both snapshots and run at least seven pinned,
   fresh processes with identical flags and inputs.
8. Refresh the commit, pristine checksum, derived checksums, retained benchmark
   artifacts, and the owning release evidence.

## Derived file checksums

These hashes identify the reviewed working snapshot. Refresh them after the
final source edit of an update checkpoint.

| File | SHA-256 |
|---|---|
| `vma.h` | `bf281c0311c9d1ce181a3919f229b2051ba467183ca4cf149d1257a9d6ab7d66` |
| `vma.hpp` | `5e260b0e8279131601226a05a93869f4e04b013e8c69028ea41deb40c8bf7ba0` |
| `vma.cpp` | `614d76e5c05f3921cbe5e193ae03e5b936cbcaf0bda9829fe5dfe34669015770` |
| `detail/config.h` | `186cf724c3f076d4a93208dfcc2a5a585f8b23ad0f084d25eefa85509968a9f9` |
| `detail/containers.h` | `f7c83426cc774599bab78960f5b54e228ee1acefa149ca5a4375aaf97492bd4d` |
| `detail/types.h` | `ac840524b991a42e0948367b55f10c89d4a7519c99d8f59b6649766fd36e69b6` |
| `detail/metadata.h` | `ad835535f953b82650238aad5edf206e4ac95acafd080fb107ea65d783b4a4a5` |
| `detail/classes.h` | `35a8bf1bed666b77edbe08c5b63a3aa13ba3ecbeb224e4034744bf263becb61b` |
| `detail/impl.h` | `a66e1bd6a3609a7a2084832b959533d74b313b3a6b50af66998103077e8d5d21` |
| `detail/allocator.h` | `7e76d0681b8c33cb632dfd201757beb5c372e41f1392ec130c353f47e8cf1fba` |
| `detail/api.h` | `9ea820005c48c08c53da4f547737ec76fb98d574b8cb870aeb3169e8918ec859` |
| `../licenses/vma.txt` | `a2f8fee7df7696fc12f5a8d292b79512d9795e0e558331c68726cd9455233593` |
