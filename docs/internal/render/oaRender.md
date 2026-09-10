# OA Rust Render

**Status:** Experimental packed Texture value; rendering and presentation sessions Planned

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Image contract:** [OA Rust Image](../image/oaImage.md)

**Donor references:** OA C++ `include/oa/runtime/texture.h`,
`include/oa/render/renderer.h`, and `include/oa/runtime/presenter.h`

Render owns Texture, Mesh, Material, Scene, Renderer, and Presenter semantics.
It consumes Image and VideoFrame without redefining those values. WSI and
native Vulkan images remain private runtime implementation details.

## Current Texture checkpoint

`oa::Texture` and `oa::render::Texture` are identity aliases for the same
value. The first backing is one checked U8 Matrix shaped
`[height, width, 4]` containing packed row-major RGBA8 pixels. Texture clones
retain storage and Texture identity; the public surface does not expose the
backing Matrix because future native sampled/storage image backing cannot be
represented honestly as a dense Matrix.

`render::texture_from_rgba8` is a synchronous, exact host upload boundary.
It rejects zero or overflowing extents and any byte count other than
`width * height * 4`. `Texture::read_rgba8` is the explicit blocking
observation boundary. `render::save_texture_file` composes that readback with
the Image module's JPEG/PNG/WebP/BMP/TGA host encoder. This replaces the donor
placement of `FnImage::saveTextureFile` with Rust subsystem ownership: Render
owns Texture completion/readback, while Image owns still-image codecs.

This packed buffer checkpoint does not claim sampled-image usage, bindless
image descriptors, filtering, native image layouts, render targets, or
presentation. Those capabilities are absent rather than represented by raw
handles or TODO-backed methods.

## Dependency order

1. Add schema-owned packed Texture clear, exact blit, and Image-to-Texture
   conversion while preserving Image and Texture graph kinds.
2. Add a private native-image backing with explicit usage, layout, queue-family,
   readiness, and consumer-retirement state.
3. Admit one headless Renderer target ring with `begin`, `submit`, explicit
   readback/abandon/collect, resize, and close transitions.
4. Add graphics pipelines and Scene/Mesh values after the VLM convention is
   consumed without local matrix fixes.
5. Add Presenter only after Engine construction accepts caller-provided WSI
   instance extensions and surface capability negotiation. Presenter borrows
   Engine and owns surface/swapchain lifecycle; Drop never presents, drains, or
   waits.
6. Compose UI and MediaPlayer over Renderer/Presenter rather than creating a
   second renderer or universal untyped player.

## Evidence

Focused tests cover the root/module type identity, exact RGBA upload/readback,
clone retention, PNG file sink, and invalid extent/range rejection on the
selected Vulkan device. Broader Render and WSI qualification remains Planned.
