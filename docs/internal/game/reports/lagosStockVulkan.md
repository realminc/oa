# moto g06 power (`lagos`) — stock Vulkan report

**Status:** Experimental evidence

**Captured:** 2026-09-10

**Related plan:** [OA Lunar Android readiness](../androidReadiness.md)

## 1. Result

ADB authorization and USB deployment access work. The connected handset is a
Motorola `moto g06 power`, device codename `lagos`, rather than the initially
recalled Moto G30.

The stock Mali driver exposes Vulkan 1.3, compute, swapchain presentation,
synchronization2, dynamic rendering, timeline semaphores, buffer device address,
FP16, INT8, and 8/16-bit storage. It does **not** satisfy the current OARS runtime
contract because the descriptor-indexing features used by OARS are absent. No
OARS APK, dispatch, renderer, ML/RL kernel, or training workload was run in this
capture.

## 2. Device identity

| Property | Observed value |
|---|---|
| Manufacturer/model | Motorola `moto g06 power` |
| Product/device | `lagos_gpe` / `lagos` |
| Android | 15, API 35 |
| ABI | `arm64-v8a` |
| SoC property | MediaTek `MT6769V/CB` |
| CPU count | 8 |
| System RAM | 3,821,080 KiB reported by `/proc/meminfo` |
| Physical display | 720 × 1640, density 280 |
| Verified boot / flash lock | `green` / locked |

The phone was USB-powered at capture time. Android reported 64% battery and
30.0 °C battery temperature. These values establish capture conditions only and
are not thermal or battery-life evidence.

## 3. Vulkan identity and limits

| Property | Observed value |
|---|---|
| Android package feature | Vulkan 1.3.0, compute, level 1 |
| Physical-device API | Vulkan 1.3.278 |
| Device | Mali-G52 MC2 |
| Vulkan vendor/device ID | `0x13b5` / `0x74021000` |
| Driver | Mali-G52 MC2, `v1.r49p1-03bet0.3a1759ee1cb57182986f58db4ad336c2` |
| Conformance | 1.3.8.0 |
| Device-local heap | 3,699,400,704 bytes, shared system memory |
| Compute queues | one family, two queues, graphics/compute/transfer, 64-bit timestamps |
| Max compute workgroup invocations | 384 |
| Max compute workgroup size | 384 × 384 × 384 |
| Max compute shared memory | 32,768 bytes |
| Max push constants | 256 bytes |
| Stock timestamp period | approximately 76.923 ns |

The Android `vkjson` command labels itself deprecated. Its report is sufficient
for reconnaissance; the future OARS Android capability executable must query and
serialize these properties directly.

## 4. OARS feature admission

| Current OARS requirement | Stock driver |
|---|---|
| Vulkan physical-device API >= 1.3 | pass |
| Hardware compute queue | pass |
| `timelineSemaphore` | pass |
| `synchronization2` | pass |
| `runtimeDescriptorArray` | **fail** |
| `descriptorBindingPartiallyBound` | **fail** |
| `descriptorBindingStorageBufferUpdateAfterBind` | **fail** |
| `descriptorBindingUpdateUnusedWhilePending` | **fail** |
| At least three update-after-bind storage descriptors | **fail**; aggregate pool limit reports zero |
| At least 16 push-constant bytes | pass; 256 bytes |

The exact current result is therefore **OARS device admission: fail**. The driver
also reports general descriptor indexing as unavailable. The failure is not a
Vulkan-version failure and must not be presented as lack of Vulkan compute.

Other relevant positive features include `bufferDeviceAddress`, shader FP16,
shader INT8, storage-buffer 8-bit access, storage-buffer 16-bit access,
`dynamicRendering`, `maintenance4`, and `VK_KHR_swapchain`.

These flags prove hardware/driver availability only. OARS currently implements
general dense F32/I32/U32 storage and has no general dense F16 operation claim.
The phone therefore needs no replacement driver to expose FP16; OARS needs a
complete F16 storage, conversion, kernel, autograd where applicable, model-file,
capability, and numerical-validation slice before the game may consume it.

## 5. Consequence for OA Lunar

This is a strong floor device for the product precisely because it exposes a
real compatibility boundary. OA Lunar can target its display and memory budget,
but current OARS cannot create a device on its stock driver.

The architecture decision must compare at least:

1. a bounded-descriptor OARS lowering that preserves the same public semantic
   graph and explicit submission model;
2. a capability-gated compatibility profile with independent kernel coverage;
3. an app-local driver route for a compatible GPU family, only after licensing,
   package-size, reliability, and shader-compiler qualification.

Do not choose among these from the feature list alone. First port a minimal
Android native capability/dispatch harness, prove a storage-buffer round trip,
inventory actual operation descriptor pressure, and identify which ML/RL and
render kernels require unavailable features.

### Driver policy

Turnip cannot be the default driver on this device. [Mesa documents
Turnip](https://docs.mesa3d.org/drivers/freedreno.html) as its Vulkan driver for
Qualcomm Adreno GPUs, while `lagos` exposes Arm Mali-G52. The OA C++ Mobile
Lab's libadrenotools/Turnip path is consequently donor evidence for Adreno
devices only.

Mesa PanVK is the corresponding research direction for Mali, but it is not a
product fallback for this handset. [Current Mesa Panfrost/PanVK
documentation](https://docs.mesa3d.org/drivers/panfrost.html) describes PanVK as
conformant on Mali-G610 and non-conformant on other GPUs; experimental devices
may refuse to load without an explicit broken-driver opt-in and may require a
newer kernel driver. That is incompatible with a default production route on
this locked stock phone. Mesa's [Android build
guidance](https://docs.mesa3d.org/android.html) also records additional rough
edges for Android drivers other than Freedreno over KGSL.

The initial OA Lunar policy is:

| GPU/driver case | Product route |
|---|---|
| This Mali-G52 phone | stock Vulkan plus an OARS bounded-descriptor compatibility route |
| Qualified Qualcomm Adreno device | stock Vulkan first; optionally isolated bundled Turnip with measured fallback behavior |
| Unknown Android GPU | stock Vulkan capability probe; fail visibly or use a separately qualified CPU path |

FP16 and driver choice remain independent. Precision selection depends on
operation coverage, numerical policy, and measurements; driver selection
depends on GPU identity and a qualified capability profile.

## 6. Capture commands

```bash
adb devices -l
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.product.device
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abilist
adb shell getprop ro.soc.manufacturer
adb shell getprop ro.soc.model
adb shell pm list features
adb shell cat /proc/meminfo
adb shell dumpsys battery
adb shell wm size
adb shell wm density
adb shell cmd gpu vkjson
```

No serial number or user data is retained in this report.
