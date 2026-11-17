# OA v0.11.3 — Rust development preview

## Summary

v0.11.3 replaces automatic Strict/Compatibility negotiation with one execution
profile per build. Strict is now named Core. `OA_BUILD_PROFILE=core|compatibility`
selects shader compilation, embedding and the Rust binding backend together;
an unset variable selects Core. Compatibility artifacts are mechanically adapted
from the same shader sources without compiling Core artifacts first.

The generated compute registry embeds 538 artifacts instead of two tables.
The complete Core catalog contains 565 SPIR-V files (17,224,848 raw bytes), and
Compatibility contains 564 (17,206,908), compared with the previous dual catalog's
1,103 files (34,309,428 bytes). These are artifact measurements, not execution
or build-time speedup claims.

Raster validation also fixes draw-parameter feature enablement and read-only
storage-buffer declarations, preserving shader arithmetic and binding layouts.

## Compatibility and migration

Core requires Vulkan 1.3 and its existing bindless feature contract. Older
hardware must explicitly build with `OA_BUILD_PROFILE=compatibility`; that build
requires Vulkan 1.2 and the bounded-binding feature contract. Runtime environment
changes cannot switch an already compiled library, and device selection never
falls back to the other profile. This changes the default hardware admission
behavior; the public semantic operation/value API remains unchanged.

Use the same build setting for Rust dependencies and Python extension builds.
Core-only video encode conversion reports a capability error in Compatibility.
Optional numeric, graphics and codec capabilities retain their runtime gates.

## Verification

The implementation passed both all-target/all-feature Clippy configurations
with warnings denied and all-features Rust suites: Core 554 tests and
Compatibility 555 tests, each with zero failures and 816 ignored tests. Eighteen
focused cases passed on Intel HD Graphics 4400 / Mesa 26.2.5 under separate core
and synchronization validation with no validation VUIDs, errors or warnings.
These cover replay, JIT, cache restore and independent raster pixel oracles.

Fresh Python binding/tooling checks ran 735 tests with zero failures and 15
skips; 76 unchanged cryptography shader-component tests passed in the earlier
broad run. Generation, formatting, inventory and whitespace checks passed.
Implementation verification used 0.11.2 package metadata; the 0.11.3 release
metadata and documentation update is checked separately without another build.

## Known limitations

Core GPU execution needs qualification on eligible hardware. The local Vulkan
1.2 device qualifies Compatibility execution and explicit Core rejection only.
GPU-assisted and cross-vendor qualification remain separate. Ignored/skipped
tests are not support claims. The repository remains Experimental.

## Upgrade

Update `oarust` and `oapython` to 0.11.3. On older hardware build with:

```sh
OA_BUILD_PROFILE=compatibility cargo build --release
OA_BUILD_PROFILE=compatibility .venv/bin/maturin develop --release
```

Caches remain profile-scoped and are rebuilt when the enabled feature identity
changes. No manual deletion of model checkpoints or runtime caches is needed.

# OA v0.11.2 — Rust development preview

## Summary

v0.11.2 lands the UI/visualization checkpoint above v0.11.1. The UI
compute compositor is replaced by a Vulkan raster pipeline:
`record_render` records one dynamic-rendering pass per frame, elements
draw in recorded order under fixed-function alpha blending, Vulkan
scissor provides clipping, and RGBA media samples through per-command
image samplers. The 14 UI compute shaders are retired.

The retained `plot` model grows log/symlog scales, error-band fill,
grouped and stacked bars, box summaries, explicit tick control, bounded
streaming series, and qualified figure file sinks (PNG/JPEG/WebP) with
byte-identical cross-process headless gallery output. `plot::FigureView`
adds pointer-anchored wheel zoom, drag pan, shift-drag box zoom, reset,
legend toggles, a crosshair tooltip, serializable `FigureViewState`, and
view-transition animation.

`Ui` gains a data table, filterable combo, color picker, multi-line text
input, menus over the popup route, generic drag-drop payloads, keyboard
and gamepad navigation, and IME candidate-rect routing. `UiStyle` token
presets are generated from `ui_style.json` (`realm_dark` base;
`realm_light` and `viewer_dark` derived). `ViewerMode::Workspace` adds a
serializable dock host over `split_pane`/`tab_bar` with tab transfer,
edge-drop split zones, and empty-leaf collapse, entered through the
`oa workspace` subcommand. Python gains the full recorded `Axes`
surface plus `FigureView` and `Workspace` bindings.

## Compatibility and migration

`UiTabBarResult` gains `dragged_index`/`drop_index`/`dropped` and
`UiTabBarState` gains drag bookkeeping; struct literal initializers
must name the new fields. `InputEvent` gains `gamepad_*` fields and
`InputEventKind` gains the `Gamepad*` variants. `Ui`, `plot`, `Viewer`,
and `Workspace` additions are otherwise additive. Rust (`oarust`),
Python (`oapython`, import `oa`), and their lockfile version metadata
agree on 0.11.2.

## Performance

No release performance claim is made. Recorded evidence is on a single
Intel Haswell iGPU under the Mesa Compatibility profile; the raster
path's per-frame cost has not been benchmarked against the retired
compute compositor.

## Verification

Local gates on the release commit: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `python3 tool/oa gen fn
--check`, `cargo check --all-targets`, the `render` host suite plus the
ignored Vulkan UI tests (`ui_workspace_*`, `ui_menu_*`, navigation, IME,
drag-drop, table, combo, color picker), Python binding tests, and the
headless gallery determinism evidence (byte-identical SHA-256 across
fresh processes, including a cold shader-cache run). Bounded `oa
workspace` and `oa view` runs open cleanly on a live Wayland session.

## Known limitations

The 3D-viewer foundation (raster roadmap stage 5) is not present.
Planar video planes remain buffer reads; render-target-backed textures
are unadmitted to `image_at`. Hands-on interactive qualification of the
viewer and workspace is pending. Evidence is single-device Intel
Haswell; no cross-vendor qualification is implied.

## Upgrade

Bump `oarust`/`oapython` to 0.11.2. `UiTabBarResult`/`UiTabBarState`
literal initializers gain fields; all other APIs are additive. Cached
`.vkcache` artifacts are disposable and rebuild as needed.

# OA v0.11.1 — Rust development preview

## Summary

v0.11.1 lands two feature checkpoints above v0.11.0. The video pipeline
begins the video-native vision roadmap (VNV-01/02/03/04/06): device-planar
YUV storage, host-planar upload, tier-1 native geometry operations (resize,
letterbox, plane stack, normalize), overlay/composite, range conversion, and
threshold/luma analysis, all schema-owned, with exactly one fused
`video::to_matrix` YUV→RGB/Matrix conversion boundary.

`nn::Empyrealm` is re-composed from a Mamba-3 parameter alias into a
Nemotron-style hybrid. `EmpyrealmConfig` schedules an interleaved stack of
Mamba-3 and causal `MultiHeadAttention` mixers, each block wrapped in
pre-norm residual wiring with a dense or routed-expert feed-forward tail and
a final RMSNorm. Inference-only single-token decode advances caller-owned
`EmpyrealmState` (one `Mamba3State` or `AttentionCache` per block) through
`Empyrealm::step`/`step_embedded`; `MultiHeadAttention::new_cache`/`step`
own the single-token key/value append path.

## Compatibility and migration

`Empyrealm::new_with_seed` now accepts `EmpyrealmConfig` instead of
`Mamba3Config`; `new_state`/`step_embedded` use `EmpyrealmState` instead of
`Mamba3State`; the former `mixer()` accessor is removed; `step`,
`layer_count`, and `attention_indices` are added. `MultiHeadAttention`
gains `new_cache` and `step`. Python `oa.ml.Empyrealm` mirrors these
changes and adds `EmpyrealmConfig`/`EmpyrealmState`. Rust (`oarust`,
namespace `oa`), Python (`oapython`, import `oa`), and their lockfile
version metadata agree on 0.11.1.

## Performance

No release performance claim is made. Recorded evidence is Haswell
exploratory rows on a single Intel iGPU under an unfixed-clock desktop
protocol; the hybrid `byte_empyrealm` posts the strongest byte-row
evaluation loss in the NLP suite (0.179619 versus Byte Transformer 0.190565
and Byte Mamba-3 0.207903) at roughly 3.4× the former composition's device
time on that workload. See `oaHaswellExploratoryBaseline.md`.

## Verification

Local gates on the release commit: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, fn/nn generator drift checks
(252 + 26 outputs byte-stable) and 45 NN generator unit tests, `cargo test
--test ml` contract tests plus the ignored hardware suite (12 Empyrealm,
attention-cache, MoE-tail, and ByteEmpyrealm tests including step-versus-scan
and cache-ownership oracles), the fresh-process 300-step Byte Empyrealm gate
(loss 0.179619, accuracy 92.9688%, exact generation, checkpoint restore),
24 hardware `convert::` video tests, and 192 Python binding tests. The
realminc/oa tag pipeline additionally builds and tests the exact tag and
verifies version-source agreement and tag ancestry on main.

## Known limitations

Video-native decode, presentation, encode-native input, P010 storage, and
compiler fusion remain gated; the video rows are Haswell-only evidence.
Empyrealm decode is inference-only and rejects active gradient recording.
Evidence is single-device Intel Haswell; no cross-vendor qualification is
implied.

## Upgrade

Bump `oarust`/`oapython` to 0.11.1. `Empyrealm` construction and decode
call sites must move to `EmpyrealmConfig`/`EmpyrealmState`; all other APIs
are unchanged. Cached `.vkcache` artifacts are disposable and rebuild as
needed.
