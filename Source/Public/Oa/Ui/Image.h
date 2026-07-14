// OaUi — image representations for display via OaUi::Image() / OaUi::ImagePlanar().
//
// Two layouts:
//
//   OaTexture     — packed RGBA8, single OaVkBuffer.  Fast display path.
//                   Load via stb_image (JPEG, PNG, BMP, TGA, HDR → u8 RGBA8).
//                   BlitRgba.slang copies it to the compose image.
//
//   OaImagePlanes — planar, one OaVkBuffer per channel, independent dtype per
//                   channel (U8 / U16 / F32 / BF16).  Better for GPU compute:
//                   channel-wise ops are coalesced; mixed precision is free.
//                   BlitPlanar.slang reads the planes and writes to compose.
//
// Typical flows:
//   File → OaTexture::LoadFile()           — JPEG/PNG display, lowest overhead
//   File → OaImagePlanes::LoadFile()       — same file, planar U8, compute-ready
//   Sensor depth → OaImagePlanes           — RGB U8 + depth U16, zero copy
//   ML output → OaImagePlanes::FromPlanes  — F32/BF16 planes from OaVkBuffer refs

#pragma once

#include <array>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Image.h>
#include <Oa/Runtime/Allocator.h>

class OaComputeEngine;


// ─── OaImageDtype ─────────────────────────────────────────────────────────────
// Per-channel storage type.  Affects how BlitPlanar reads and normalizes values.

enum class OaImageDtype : OaU8 {
	U8   = 0,  // uint8   [0, 255]   → [0.0, 1.0]
	U16  = 1,  // uint16  [0, 65535] → [0.0, 1.0]
	F32  = 2,  // float32  passed through as-is
	BF16 = 3,  // bfloat16 decoded to float32 as-is
};


// ─── OaTexture ────────────────────────────────────────────────────────────────
// Packed RGBA8 device-local buffer.  Pass BindlessIndex() to OaUi::Image().

struct OaTexture {
	OaVkBuffer DeviceBuf;
	OaI32      Width  = 0;
	OaI32      Height = 0;

	// ── Image-backed view (UnifiedExecutionArchitecture.md §3.5, Step 3b.3) ─────────────────────
	// Non-owning handles populated by OaContext::RecordAcquire to wrap a
	// swapchain image as an OaTexture. When Image != nullptr, this texture
	// references a VkImage (typically owned by OaSwapchain) rather than a
	// buffer-backed allocation in the bindless heap. Layout tracks the
	// current VkImageLayout across record APIs so RecordPresent can emit the
	// correct barrier. Typed as void*/int to keep <vulkan/vulkan.h> out of
	// this header, matching OaVkBuffer's convention in Allocator.h.
	void* Image  = nullptr;  // VkImage
	void* View   = nullptr;  // VkImageView
	OaI32 Format = 0;        // VkFormat (0 = VK_FORMAT_UNDEFINED)
	OaI32 Layout = 0;        // VkImageLayout (0 = VK_IMAGE_LAYOUT_UNDEFINED)

	[[nodiscard]] OaU32 BindlessIndex() const noexcept { return DeviceBuf.BindlessIndex; }
	[[nodiscard]] bool  IsImageBacked() const noexcept { return Image != nullptr; }
	[[nodiscard]] bool  IsValid()       const noexcept {
		return DeviceBuf.Buffer != nullptr or Image != nullptr;
	}

	// Decode any stb_image-supported format (JPEG, PNG, BMP, TGA, HDR) and
	// upload to GPU. Output is always RGBA8. Synchronous (returns after transfer).
	[[nodiscard]] static OaResult<OaTexture> LoadFile(
		OaComputeEngine& InRt,
		OaStringView       InPath);

	// Upload caller-supplied RGBA8 pixels (W*H*4 bytes). Synchronous.
	[[nodiscard]] static OaResult<OaTexture> FromPixels(
		OaComputeEngine& InRt,
		OaSpan<const OaU8> InRgba,
		OaI32              InW,
		OaI32              InH);

	// Free the device buffer and deregister from the bindless heap.
	// Must be called before engine shutdown. Safe to call on a default-constructed
	// (invalid) OaTexture — becomes a no-op.
	void Destroy(OaComputeEngine& InRt);

	// ── OaMatrix bridge ───────────────────────────────────────────────
	// Convert RGBA8 display buffer → [1, 4, H, W] F32 NCHW OaMatrix.
	[[nodiscard]] OaMatrix ToMatrix() const;

	// Upload a [1, C, H, W] or [1, 4, H, W] OaMatrix back into a packed
	// RGBA8 OaTexture for display. Values are clamped to [0, 1] and scaled to u8.
	[[nodiscard]] static OaResult<OaTexture> FromMatrix(
		OaComputeEngine& InRt,
		const OaMatrix& InMatrix);

	// ── Phase 4: OaImage bridge ─────────────────────────────────────────
	// Convert RGBA8 display buffer → OaImage with semantic metadata.
	// Returns OaImage with Nhwc layout and Rgba format.
	[[nodiscard]] OaImage ToImage() const;

	// Create OaTexture from OaImage for display.
	// Converts to RGBA8 packed format regardless of input format/layout.
	[[nodiscard]] static OaResult<OaTexture> FromImage(
		OaComputeEngine& InRt,
		const OaImage& InImage);

	// ── Operator overloading ──────────────────────────────────────────
	// Note: These operators work through ToMatrix()/FromMatrix() bridge.
	// For performance-critical code, consider working directly with OaMatrix.
	
	// Element-wise operations (requires same dimensions)
	[[nodiscard]] OaTexture operator+(const OaTexture& InOther) const;
	[[nodiscard]] OaTexture operator-(const OaTexture& InOther) const;
	[[nodiscard]] OaTexture operator*(const OaTexture& InOther) const;
	[[nodiscard]] OaTexture operator/(const OaTexture& InOther) const;
	
	// Scalar operations
	[[nodiscard]] OaTexture operator+(OaF32 InScalar) const;
	[[nodiscard]] OaTexture operator-(OaF32 InScalar) const;
	[[nodiscard]] OaTexture operator*(OaF32 InScalar) const;
	[[nodiscard]] OaTexture operator/(OaF32 InScalar) const;
	
	// Unary negation
	[[nodiscard]] OaTexture operator-() const;
};


// ─── OaImagePlanes ────────────────────────────────────────────────────────────
// Planar image: one device-local OaVkBuffer per channel, independent dtype.
//
// Layout: each plane is a flat array of W*H elements in row-major order.
// Channels:  1=gray, 2=gray+alpha, 3=RGB, 4=RGBA or RGB+depth.
// Missing alpha plane (ChannelCount < 4) defaults to 1.0 in the blit shader.
//
// Pass to OaUi::ImagePlanar().

static constexpr OaU32 kOaImageMaxPlanes = 4;

struct OaImagePlanes {
	std::array<OaVkBuffer,   kOaImageMaxPlanes> Planes = {};
	std::array<OaImageDtype, kOaImageMaxPlanes> Dtypes = {};
	OaI32        Width        = 0;
	OaI32        Height       = 0;
	OaU8         ChannelCount = 0;

	[[nodiscard]] OaU32 PlaneIndex(OaU32 InChannel) const noexcept {
		return Planes[InChannel].BindlessIndex;
	}
	[[nodiscard]] bool IsValid() const noexcept {
		return ChannelCount > 0 and Width > 0 and Height > 0;
	}

	// Decode any stb_image-supported file to planar U8.
	// 1-channel files → gray (ChannelCount=1).
	// 3-channel files → RGB (ChannelCount=3).
	// 4-channel files → RGBA (ChannelCount=4).
	// HDR (.hdr / .exr) → planar F32.  Synchronous (transfer completes on return).
	[[nodiscard]] static OaResult<OaImagePlanes> LoadFile(
		OaComputeEngine& InRt,
		OaStringView       InPath);

	// Upload caller-provided planes.  InPlanes and InDtypes must have the same
	// length (1–4).  Each span is W*H * sizeof(dtype) bytes.  Synchronous.
	[[nodiscard]] static OaResult<OaImagePlanes> FromPlanes(
		OaComputeEngine&                 InRt,
		OaSpan<const OaSpan<const OaU8>>   InPlanes,
		OaSpan<const OaImageDtype>         InDtypes,
		OaI32                              InW,
		OaI32                              InH);

	// Free all plane buffers and deregister from the bindless heap.
	// Safe to call on invalid / partially-initialised instances.
	void Destroy(OaComputeEngine& InRt);

	// ── OaMatrix bridge ───────────────────────────────────────────────
	// Convert planar image → [1, C, H, W] F32 NCHW OaMatrix.
	// Each plane becomes a channel; dtype is promoted to F32.
	[[nodiscard]] OaMatrix ToMatrix() const;

	// Wrap a [1, C, H, W] OaMatrix as OaImagePlanes without copying.
	// One OaVkBuffer slice per channel; dtype is inferred from matrix dtype.
	[[nodiscard]] static OaResult<OaImagePlanes> FromMatrix(
		OaComputeEngine& InRt,
		const OaMatrix& InMatrix);

	// ── Operator overloading ──────────────────────────────────────────
	// Note: These operators work through ToMatrix()/FromMatrix() bridge.
	// For performance-critical code, consider working directly with OaMatrix.
	
	// Element-wise operations (requires same dimensions)
	[[nodiscard]] OaImagePlanes operator+(const OaImagePlanes& InOther) const;
	[[nodiscard]] OaImagePlanes operator-(const OaImagePlanes& InOther) const;
	[[nodiscard]] OaImagePlanes operator*(const OaImagePlanes& InOther) const;
	[[nodiscard]] OaImagePlanes operator/(const OaImagePlanes& InOther) const;
	
	// Scalar operations
	[[nodiscard]] OaImagePlanes operator+(OaF32 InScalar) const;
	[[nodiscard]] OaImagePlanes operator-(OaF32 InScalar) const;
	[[nodiscard]] OaImagePlanes operator*(OaF32 InScalar) const;
	[[nodiscard]] OaImagePlanes operator/(OaF32 InScalar) const;

	// Unary negation
	[[nodiscard]] OaImagePlanes operator-() const;
};


// ─── OaUiImage ────────────────────────────────────────────────────────────────
//
// ImGui-binding adapter over OaTexture.
//
// OaTexture is the universal GPU resource (FinalGlue layer, no UI awareness).
// OaUiImage takes an OaTexture and produces an ImTextureID for ImGui's Vulkan
// backend: owns the descriptor set + sampler + image view that the backend
// needs to display the texture. The user's code only ever touches OaTexture
// (resource ownership) and OaUiImage (display) — no raw ImTextureID leaks.
//
// Usage:
//   OaTexture tex = OaFnImage::LoadFile(...);   // resource
//   OaUiImage uiImg(tex);                       // bind to ImGui
//   ui.Image(uiImg, {.FitMode = OaUi::Fit::Letterbox});
//   plot.Imshow(uiImg);
//   ImGui::Image(uiImg.ImGuiHandle(), ImVec2(w, h));
//
// Phase 2 (compute-shader widgets, no ImGui): this type either gets deleted
// (widgets read OaTexture's Bindless ID directly — no adapter needed) or
// becomes a deprecated alias. OaTexture's name is preserved either way.
//
// Lifecycle: ImGui descriptor pool is engine-owned (ImGuiPool_ on
// OaGraphicsEngine). Registration on construction, deferred-free
// unregistration on destruction (frame-in-flight may still reference the
// descriptor set — same dance ImGui's font texture already does).

class OaGraphicsEngine;

class OaUiImage {
public:
	OaUiImage() = default;

	// Construct from any OaTexture. Eagerly allocates the ImGui binding
	// (sampler + image view + descriptor set against the engine's ImGui pool)
	// and caches the resulting ImTextureID.
	explicit OaUiImage(const OaTexture& InTex);
	explicit OaUiImage(OaTexture&& InTex);

	OaUiImage(const OaUiImage&)            = delete;
	OaUiImage& operator=(const OaUiImage&) = delete;
	OaUiImage(OaUiImage&&) noexcept;
	OaUiImage& operator=(OaUiImage&&) noexcept;

	// Unregisters the ImGui binding (deferred to end of current frame).
	~OaUiImage();

	// Pass-through accessors.
	[[nodiscard]] const OaTexture& Texture()       const noexcept { return Tex_; }
	[[nodiscard]] OaU32            Width()         const noexcept { return static_cast<OaU32>(Tex_.Width);  }
	[[nodiscard]] OaU32            Height()        const noexcept { return static_cast<OaU32>(Tex_.Height); }
	[[nodiscard]] OaU32            BindlessIndex() const noexcept { return Tex_.BindlessIndex(); }
	[[nodiscard]] bool             IsValid()       const noexcept { return Tex_.IsValid() and ImGuiHandle_ != nullptr; }

	// ImGui-side handle. Stable for this OaUiImage's lifetime. Pass to
	// ImGui::Image() / OaUi::Image() / OaPlot::Imshow(). Typed as void* to
	// avoid pulling in <imgui.h> from this header — internally it's an
	// ImTextureID (= VkDescriptorSet for the ImGui Vulkan backend).
	[[nodiscard]] void* ImGuiHandle() const noexcept { return ImGuiHandle_; }

private:
	OaTexture Tex_;
	// Vulkan handles, opaque to keep <vulkan/vulkan.h> out of this header.
	// Created during construction, released at destruction (deferred-free).
	void*     DescriptorSet_ = nullptr;  // VkDescriptorSet (ImGui pool)
	void*     Sampler_       = nullptr;  // VkSampler (Linear, ClampToEdge)
	void*     View_          = nullptr;  // VkImageView for sampling
	void*     ImGuiHandle_   = nullptr;  // ImTextureID — equals DescriptorSet_ for imgui_impl_vulkan
};
