# OA Lunar Android development readiness

**Status:** Research

**Audited:** 2026-09-10; physical-device connection updated 2026-09-10

**Product plan:** [OA Lunar](oaLunar.md)

**Platform guidance:** [Android Vulkan native engine support](https://developer.android.com/games/develop/vulkan/native-engine-support),
[Android GameActivity](https://developer.android.com/games/agdk/game-activity/get-started)

## 1. Result

The workstation has a usable Android C++ development foundation: Android Studio,
SDK, NDK, ADB, Java, CMake, Ninja, shader tools, and the existing OA C++ Mobile
Lab Gradle wrapper are present. It is not yet a complete Rust/OARS Android
environment. The Rust ARM64 Android target and `cargo-ndk` are absent, and OARS
has no Android application or native-library packaging target. A physical
Motorola phone was subsequently connected and authorized; its report is
[moto g06 power stock Vulkan](reports/lagosStockVulkan.md).

The connected phone is a `moto g06 power` (`lagos`), not the initially recalled
Moto G30. It is a valuable minimum-performance qualification target, not yet a
supported device. Its identity, OS, ABI, and stock Vulkan capability baseline
have now been measured. Kernel correctness, memory pressure, thermals, sustained
behavior, and complete Android lifecycle behavior remain unqualified.

## 2. Workstation evidence

| Component | Audited state |
|---|---|
| Android SDK | `/home/empyrealm/Android/Sdk`; `ANDROID_HOME` and `ANDROID_SDK_ROOT` resolve to it |
| ADB | 1.0.41, Platform Tools 37.0.0 |
| Android Studio | `/home/empyrealm/Apps/android-studio`; bundled OpenJDK 21.0.10 |
| SDK platforms | Android 36 revision 2 |
| build-tools | 35.0.0 and 36.0.0 |
| NDK | 29.0.14206865; `ANDROID_NDK_HOME` resolves to it |
| ARM64 NDK compiler | Clang 21.0.0; `aarch64-unknown-linux-android33` invocation succeeds |
| Native build tools | CMake 4.4.3 and Ninja 1.13.2 on the host; SDK CMake 4.1.2 installed |
| Shader tools | Slang 2026.5; SPIR-V Tools 2026.3 from Vulkan SDK 1.4.357 |
| OA C++ Android harness | `sdk/android/mobilelab` exists in the OA C++ repository; its cached Gradle 8.13 wrapper starts offline with JDK 21 |
| USB access rules | system Android udev rules include Motorola vendor `22b8` and `uaccess`; permission remains unproved until the handset connects |

Neither a global Gradle executable nor `ANDROID_NDK_ROOT` is required by the
audited OA Mobile Lab: its repository wrapper and `ANDROID_NDK_HOME` are present.
No `local.properties` was found in OARS or the Mobile Lab; the current exported
SDK variables provide discovery, while a future Android consumer may generate an
untracked local file if its build requires one.

## 3. Missing Rust and OARS pieces

| Gap | Consequence | Closure proof |
|---|---|---|
| Rust target `aarch64-linux-android` is not installed | Cargo cannot currently produce the phone's Rust artifact | target appears in `rustup target list --installed`; minimal ARM64 library links with NDK |
| `cargo-ndk` is not installed | no standardized Cargo-to-NDK ABI/output workflow is available | pinned tool version builds `arm64-v8a` through the chosen project command |
| OARS has no Gradle project, Android manifest, GameActivity shell, or Android packaging target | there is no OARS APK to install | clean consumer build, install, launch, pause/resume, surface recreation, and uninstall on a phone |
| OARS has no Android Presenter/WSI route | compute work cannot establish game presentation | one Engine owns compute and render services while Presenter survives surface loss |
| OARS currently requests Vulkan 1.3 and rejects a physical-device API below 1.3 | an otherwise useful Vulkan 1.1 driver with equivalent extensions still fails before feature negotiation | device report plus an explicit, tested core-or-extension capability policy |

The final item is a live source constraint in
`src/rs/runtime/vk/instance.rs` and `src/rs/runtime/vk/physical.rs`, not a guess
about the Moto. Android platform version does not prove physical-driver Vulkan
support; the native device probe is authoritative.

## 4. Existing donor evidence

The OA C++ repository already contains `sdk/android/mobilelab` and dated Android
reports. The audited Redmi `creek` handset uses Qualcomm SM6225/Adreno 610 but is
not the Moto G30:

- its stock driver reports a Vulkan 1.1 physical device and passes a basic
  storage-buffer compute dispatch, but fails the C++ `ModernCompute` profile;
- its app-local Turnip route exposes the required extension-backed capabilities
  and passes that profile and basic dispatch;
- those results neither qualify OARS nor prove game rendering, ML/RL kernels,
  sustained thermals, or the Moto's stock driver.

This evidence makes an Adreno 610 handset a high-value compatibility target. It
does not justify weakening the OARS runtime blindly or bundling a replacement
driver before stock-driver measurement and licensing/product review.

## 5. First physical-device session

On the phone, enable Developer options and USB debugging, select a data-capable
USB mode, connect with a data cable, accept the host RSA prompt, and keep the
screen unlocked for the first authorization. Then capture:

```bash
adb devices -l
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.product.device
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abilist
adb shell cmd gpu vkjson
```

If ADB reports `unauthorized`, accept or revoke/recreate the USB-debugging
authorization on the phone. If the phone is absent from both `adb devices -l`
and `lsusb`, test the cable, port, USB mode, and phone settings before changing
software. If it appears in `lsusb` but ADB reports a permission error, inspect
the applied udev rule and active-seat access before changing groups.

The first report records exact device identity, build fingerprint, boot state,
Android API, ABI, Vulkan loader/physical-device/driver/conformance versions,
extensions, feature structures, limits, queue families, memory heaps, and the
result of a bounded storage-buffer dispatch. Do not run long training or kernel
benchmarks until this gate passes.

## 6. Optimization role

Use the Moto as the floor target for bounded interactive workloads:

1. correctness and lifecycle before performance;
2. stock-driver CPU physics plus policy inference baseline;
3. representative ML/RL kernels with independent CPU or donor oracles;
4. fixed warmup/cooldown and at least seven fresh-process measurements;
5. frame-time percentiles, inference latency, memory, battery/thermal status,
   throttling, and fallback counters;
6. short on-device training only after inference and sustained-load gates pass.

Optimization results name the exact APK/OARS commits, model and workload,
precision, Android build, phone, GPU driver, compiler/shader provenance, power
state, temperature state, samples, median, and spread. The handset can define the
initial floor, but broad Android support requires additional stock-driver devices
from other GPU families.

## 7. Reproduction commands

The 2026-09-10 host snapshot was checked with:

```bash
adb version
adb devices -l
sdkmanager --list_installed
rustc --version
cargo --version
rustup target list --installed
cargo ndk --version
/home/empyrealm/Android/Sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang --version
cd /home/empyrealm/Code/GitHub/oa/sdk/android/mobilelab
./gradlew --offline --no-daemon --version
```

The first observation was an empty `adb devices -l` list. After USB debugging
was enabled and the host RSA prompt accepted, ADB reported one authorized
`moto_g06_power` device over USB. The resulting physical-device capability
snapshot is recorded separately so this workstation report does not become the
owner of volatile device evidence.
