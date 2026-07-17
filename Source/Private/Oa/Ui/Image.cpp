// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>

#include <Oa/Ui/Image.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>

#include "../../../ThirdParty/stb/stb_image.h"

#include <cstring>

// ─── helpers ──────────────────────────────────────────────────────────────────

static OaResult<OaVkBuffer> UploadToDevice(
	OaComputeEngine& InRt,
	const void* InData,
	OaU64 InBytes)
{
	auto stagingRes = InRt.AllocBuffer(InBytes);
	if (!stagingRes) return stagingRes.GetStatus();
	OaVkBuffer staging = *stagingRes;
	std::memcpy(staging.MappedPtr, InData, InBytes);

	auto deviceRes = InRt.AllocBufferDevice(InBytes);
	if (!deviceRes) {
		InRt.FreeBuffer(staging);
		return deviceRes.GetStatus();
	}
	OaVkBuffer device = *deviceRes;

	if (auto s = InRt.CopyBufferAsync(staging, device, InBytes); !s.IsOk()) {
		InRt.FreeBuffer(staging);
		InRt.FreeBuffer(device);
		return s;
	}
	if (auto s = InRt.WaitTransfer(); !s.IsOk()) {
		InRt.FreeBuffer(staging);
		InRt.FreeBuffer(device);
		return s;
	}

	InRt.FreeBuffer(staging);
	InRt.RegisterBuffer(device);
	return device;
}


// ─── OaTexture ────────────────────────────────────────────────────────────────

OaResult<OaTexture> OaTexture::LoadFile(OaComputeEngine& InRt, OaStringView InPath) {
	int w = 0, h = 0, ch = 0;
	stbi_uc* px = stbi_load(InPath.data(), &w, &h, &ch, 4);
	if (!px) {
		OA_LOG_ERROR(OaLogComponent::App, "OaTexture::LoadFile: stb_image failed: %s (%s)",
			InPath.data(), stbi_failure_reason());
		return OaStatus::Error("stb_image load failed");
	}

	OaU64 bytes = static_cast<OaU64>(w) * h * 4;
	auto res = UploadToDevice(InRt, px, bytes);
	stbi_image_free(px);
	if (!res) return res.GetStatus();

	OaTexture img;
	img.DeviceBuf = *res;
	img.Width     = w;
	img.Height    = h;
	OA_LOG_INFO(OaLogComponent::App, "OaTexture: loaded %s (%dx%d)", InPath.data(), w, h);
	return img;
}

OaResult<OaTexture> OaTexture::FromPixels(
	OaComputeEngine& InRt,
	OaSpan<const OaU8> InRgba,
	OaI32 InW,
	OaI32 InH)
{
	OaU64 expected = static_cast<OaU64>(InW) * InH * 4;
	if (InRgba.Size() < expected) {
		return OaStatus::Error("FromPixels: span too small");
	}

	auto res = UploadToDevice(InRt, InRgba.Data(), expected);
	if (!res) return res.GetStatus();

	OaTexture img;
	img.DeviceBuf = *res;
	img.Width     = InW;
	img.Height    = InH;
	return img;
}

void OaTexture::Destroy(OaComputeEngine& InRt) {
	if (DeviceOwner) {
		DeviceOwner.Reset();
	} else if (DeviceBuf.Buffer) {
		InRt.DeregisterBuffer(DeviceBuf);
		InRt.FreeBuffer(DeviceBuf);
	}
	DeviceBuf = {};
	// Image-backed textures are non-owning views (the swapchain or other
	// upstream owns the VkImage/VkImageView). Just drop the references.
	Image  = nullptr;
	View   = nullptr;
	Layout = 0;
	Width  = 0;
	Height = 0;
}


// ─── OaImagePlanes ────────────────────────────────────────────────────────────

OaResult<OaImagePlanes> OaImagePlanes::LoadFile(OaComputeEngine& InRt, OaStringView InPath) {
	int w = 0, h = 0, ch = 0;
	// Query channel count first
	stbi_info(InPath.data(), &w, &h, &ch);
	if (w == 0 || h == 0) {
		// Try loading to get dimensions
		ch = 4;
	}

	// Determine if HDR
	bool isHdr = stbi_is_hdr(InPath.data()) != 0;
	OaImagePlanes planes;

	if (isHdr) {
		float* px = stbi_loadf(InPath.data(), &w, &h, &ch, 0);
		if (!px) {
			OA_LOG_ERROR(OaLogComponent::App, "OaImagePlanes::LoadFile: HDR load failed: %s", InPath.data());
			return OaStatus::Error("stb_image HDR load failed");
		}
		OaU32 nCh = static_cast<OaU32>(ch);
		OaU64 planeBytes = static_cast<OaU64>(w) * h * sizeof(float);

		OaVec<float> tmp(static_cast<OaU64>(w) * h);
		for (OaU32 c = 0; c < nCh and c < kOaImageMaxPlanes; ++c) {
			for (OaI64 i = 0; i < static_cast<OaI64>(w) * h; ++i) {
				tmp[static_cast<OaU64>(i)] = px[static_cast<OaU64>(i) * nCh + c];
			}
			auto res = UploadToDevice(InRt, tmp.Data(), planeBytes);
			if (!res) { stbi_image_free(px); planes.Destroy(InRt); return res.GetStatus(); }
			planes.Planes[c] = *res;
			planes.Dtypes[c] = OaImageDtype::F32;
		}
		stbi_image_free(px);
		planes.Width        = w;
		planes.Height       = h;
		planes.ChannelCount = static_cast<OaU8>(std::min(nCh, kOaImageMaxPlanes));
	} else {
		stbi_uc* px = stbi_load(InPath.data(), &w, &h, &ch, 0);
		if (!px) {
			OA_LOG_ERROR(OaLogComponent::App, "OaImagePlanes::LoadFile: load failed: %s", InPath.data());
			return OaStatus::Error("stb_image load failed");
		}
		OaU32 nCh = static_cast<OaU32>(ch);
		OaU64 planeBytes = static_cast<OaU64>(w) * h;

		OaVec<OaU8> tmp(static_cast<OaU64>(w) * h);
		for (OaU32 c = 0; c < nCh and c < kOaImageMaxPlanes; ++c) {
			for (OaI64 i = 0; i < static_cast<OaI64>(w) * h; ++i) {
				tmp[static_cast<OaU64>(i)] = px[static_cast<OaU64>(i) * nCh + c];
			}
			auto res = UploadToDevice(InRt, tmp.Data(), planeBytes);
			if (!res) { stbi_image_free(px); planes.Destroy(InRt); return res.GetStatus(); }
			planes.Planes[c] = *res;
			planes.Dtypes[c] = OaImageDtype::U8;
		}
		stbi_image_free(px);
		planes.Width        = w;
		planes.Height       = h;
		planes.ChannelCount = static_cast<OaU8>(std::min(nCh, kOaImageMaxPlanes));
	}

	OA_LOG_INFO(OaLogComponent::App, "OaImagePlanes: loaded %s (%dx%d, %u ch)",
		InPath.data(), w, h, planes.ChannelCount);
	return planes;
}

OaResult<OaImagePlanes> OaImagePlanes::FromPlanes(
	OaComputeEngine& InRt,
	OaSpan<const OaSpan<const OaU8>> InPlanes,
	OaSpan<const OaImageDtype>       InDtypes,
	OaI32 InW,
	OaI32 InH)
{
	if (InPlanes.Size() == 0 or InPlanes.Size() > kOaImageMaxPlanes) {
		return OaStatus::Error("FromPlanes: channel count must be 1-4");
	}
	if (InPlanes.Size() != InDtypes.Size()) {
		return OaStatus::Error("FromPlanes: planes/dtypes size mismatch");
	}

	OaImagePlanes planes;
	planes.Width        = InW;
	planes.Height       = InH;
	planes.ChannelCount = static_cast<OaU8>(InPlanes.Size());

	for (OaU32 c = 0; c < InPlanes.Size(); ++c) {
		auto res = UploadToDevice(InRt, InPlanes[c].Data(), InPlanes[c].Size());
		if (!res) { planes.Destroy(InRt); return res.GetStatus(); }
		planes.Planes[c] = *res;
		planes.Dtypes[c] = InDtypes[c];
	}
	return planes;
}

// ─── OaTexture OaMatrix bridge ──────────────────────────────────────────

OaMatrix OaTexture::ToMatrix() const
{
	// RGBA8 packed buffer → [1, 4, H, W] F32 NCHW.
	// The device buffer holds W*H*4 bytes in row-major interleaved RGBA order.
	// We reinterpret via OaFnMatrix::FromBytes and let the caller convert/crop
	// channels as needed (e.g. drop alpha with OaFnMatrix::Slice).
	const OaU64 bytes = static_cast<OaU64>(Width) * Height * 4 * sizeof(OaF32);
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(static_cast<const OaU8*>(DeviceBuf.MappedPtr), bytes),
		OaMatrixShape{1, 4, static_cast<OaI64>(Height), static_cast<OaI64>(Width)},
		OaScalarType::Float32);
}

OaResult<OaTexture> OaTexture::FromMatrix(
	OaComputeEngine& InRt,
	const OaMatrix& InMatrix)
{
	(void)InRt; // allocation is owned by the active context's matrix pool
	const auto shape = InMatrix.GetShape();
	if (shape.Rank != 4 || shape[0] != 1) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: expected [1,C,H,W] matrix");
	}
	const OaI32 channels = static_cast<OaI32>(shape[1]);
	if (channels != 1 && channels != 3 && channels != 4) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: channel count must be 1, 3 or 4");
	}
	if (InMatrix.GetDtype() != OaScalarType::Float32
		&& InMatrix.GetDtype() != OaScalarType::BFloat16
		&& InMatrix.GetDtype() != OaScalarType::Float16) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: expected a floating-point matrix");
	}
	const OaI32 h = static_cast<OaI32>(shape[2]);
	const OaI32 w = static_cast<OaI32>(shape[3]);
	if (w <= 0 || h <= 0) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: dimensions must be positive");
	}

	const OaI64 pixelCount = static_cast<OaI64>(w) * h;
	// UInt32 is exactly one packed RGBA8 pixel. It also gives the graph a shared
	// allocation owner, avoiding a raw-buffer lifetime escape.
	auto packed = OaFnMatrix::Empty(
		OaMatrixShape{pixelCount}, OaScalarType::UInt32);
	struct Push {
		OaU32 Channels;
		OaU32 Height;
		OaU32 Width;
	} push{
		static_cast<OaU32>(channels),
		static_cast<OaU32>(h),
		static_cast<OaU32>(w),
	};
	OaBufferAccess access[] = {
		OaBufferAccess::Read, OaBufferAccess::Write};
	auto& ctx = OaContext::GetDefault();
	ctx.Add("MatrixToRgba8", {&InMatrix, &packed}, access,
		&push, sizeof(push),
		(static_cast<OaU32>(pixelCount) + 255U) / 256U);

	OaTexture texture;
	texture.DeviceBuf = packed.GetVkBuffer();
	texture.DeviceOwner = packed.VkBuf_;
	texture.Width = w;
	texture.Height = h;
	return texture;
}

// ─── Phase 4: OaImage bridge ───────────────────────────────────────────────

OaImage OaTexture::ToImage() const
{
	// Convert RGBA8 packed buffer → OaImage with semantic metadata.
	// Returns OaImage with Nhwc layout and Rgba format.
	OaMatrix mat = ToMatrix();
	return OaImage(std::move(mat), OaImageLayout::Nhwc, OaImageFormat::Rgba);
}

OaResult<OaTexture> OaTexture::FromImage(
	OaComputeEngine& InRt,
	const OaImage& InImage)
{
	// Convert OaImage to OaTexture for display.
	// Convert to RGBA8 packed format regardless of input format/layout.
	// For now, delegate to the OaMatrix bridge.
	return FromMatrix(InRt, InImage.AsMatrix());
}

// ─── OaImagePlanes OaMatrix bridge ────────────────────────────────────────

OaMatrix OaImagePlanes::ToMatrix() const
{
	if (!IsValid()) return OaMatrix();

	// Stack channel planes into [1, C, H, W] F32 tensor via OaFnMatrix::Concat.
	OaVec<OaMatrix> channels;
	channels.Reserve(ChannelCount);
	const OaU64 planeBytes = static_cast<OaU64>(Width) * Height * sizeof(OaF32);
	const OaMatrixShape planeShape = OaMatrixShape{1, 1, static_cast<OaI64>(Height), static_cast<OaI64>(Width)};
	for (OaU8 c = 0; c < ChannelCount; ++c) {
		channels.PushBack(OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(static_cast<const OaU8*>(Planes[c].MappedPtr), planeBytes),
			planeShape,
			OaScalarType::Float32));
	}
	return OaFnMatrix::Concat(OaSpan<OaMatrix>(channels.Data(), channels.Size()), 1);
}

OaResult<OaImagePlanes> OaImagePlanes::FromMatrix(
	OaComputeEngine& InRt,
	const OaMatrix& InMatrix)
{
	const auto shape = InMatrix.GetShape();
	if (shape.Rank < 4) {
		return OaStatus::Error("OaImagePlanes::FromMatrix: expected [1,C,H,W] matrix");
	}
	const OaI32 c = static_cast<OaI32>(shape[1]);
	const OaI32 h = static_cast<OaI32>(shape[2]);
	const OaI32 w = static_cast<OaI32>(shape[3]);
	if (c < 1 or c > static_cast<OaI32>(kOaImageMaxPlanes)) {
		return OaStatus::Error("OaImagePlanes::FromMatrix: channel count out of range");
	}

	OaImagePlanes planes;
	planes.Width        = w;
	planes.Height       = h;
	planes.ChannelCount = static_cast<OaU8>(c);

	// Slice each channel from the NCHW matrix and share its underlying VkBuffer.
	// The OaImagePlanes does not own these buffers — lifetime is tied to InMatrix.
	// TODO: When OaMatrix::GetChannelBuffer(c) is available, use that.
	// For now, store the whole-matrix buffer in slot 0 and mark others invalid.
	for (OaI32 i = 0; i < c; ++i) {
		planes.Planes[i] = InMatrix.GetVkBuffer();
		planes.Dtypes[i] = (InMatrix.GetDtype() == OaScalarType::BFloat16)
			? OaImageDtype::BF16 : OaImageDtype::F32;
	}
	return planes;
}

void OaImagePlanes::Destroy(OaComputeEngine& InRt) {
	for (OaU32 c = 0; c < kOaImageMaxPlanes; ++c) {
		if (Planes[c].Buffer) {
			InRt.DeregisterBuffer(Planes[c]);
			InRt.FreeBuffer(Planes[c]);
		}
	}
	Width = Height = 0;
	ChannelCount = 0;
}


// ============================================================================
// OaTexture operator overloads
// ============================================================================

OaTexture OaTexture::operator+(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() + InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() - InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator*(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() * InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator/(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() / InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator+(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() + InScalar;
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() - InScalar;
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator*(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() * InScalar;
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator/(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() / InScalar;
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-() const {
	OaMatrix result = -this->ToMatrix();
	auto img = OaTexture::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaTexture{};
}

// ============================================================================
// OaImagePlanes operator overloads
// ============================================================================

OaImagePlanes OaImagePlanes::operator+(const OaImagePlanes& InOther) const {
	OaMatrix result = this->ToMatrix() + InOther.ToMatrix();
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator-(const OaImagePlanes& InOther) const {
	OaMatrix result = this->ToMatrix() - InOther.ToMatrix();
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator*(const OaImagePlanes& InOther) const {
	OaMatrix result = this->ToMatrix() * InOther.ToMatrix();
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator/(const OaImagePlanes& InOther) const {
	OaMatrix result = this->ToMatrix() / InOther.ToMatrix();
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator+(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() + InScalar;
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator-(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() - InScalar;
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator*(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() * InScalar;
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator/(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() / InScalar;
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}

OaImagePlanes OaImagePlanes::operator-() const {
	OaMatrix result = -this->ToMatrix();
	auto img = OaImagePlanes::FromMatrix(*OaContext::GetDefault().GetEngine(), result);
	return img.IsOk() ? *img : OaImagePlanes{};
}


// ═══════════════════════════════════════════════════════════════════════════
// OaUiImage — ImGui-binding adapter over OaTexture
// ───────────────────────────────────────────────────────────────────────────
// Step 3b SKELETON. Real impl needs to:
//   * call ImGui_ImplVulkan_AddTexture(sampler, view, layout) to register a
//     descriptor set against ImGuiPool_ on OaGraphicsEngine
//   * allocate a VkImageView for the underlying OaTexture (if not already
//     image-backed — today OaTexture is buffer-backed, so this requires
//     either a host-side memcpy → VkImage or a compute readback dispatch)
//   * arrange deferred-free at end of current frame
// All of that lands with the engine-owned compose image in Step 3c. Until
// then OaUiImage is constructible (so headers and tests link) but doesn't
// produce a valid ImTextureID — IsValid() returns false.

#include <Oa/Core/Log.h>


OaUiImage::OaUiImage(const OaTexture& InTex)
	: Tex_(InTex)
{
	// TODO(Step 3b.5): register descriptor set with ImGui's Vulkan backend.
	// ImGuiHandle_ stays null for now; IsValid() returns false.
}

OaUiImage::OaUiImage(OaTexture&& InTex)
	: Tex_(std::move(InTex))
{
	// TODO(Step 3b.5): register descriptor set with ImGui's Vulkan backend.
}

OaUiImage::OaUiImage(OaUiImage&& InOther) noexcept
	: Tex_(std::move(InOther.Tex_))
	, DescriptorSet_(InOther.DescriptorSet_)
	, Sampler_(InOther.Sampler_)
	, View_(InOther.View_)
	, ImGuiHandle_(InOther.ImGuiHandle_)
{
	InOther.DescriptorSet_ = nullptr;
	InOther.Sampler_       = nullptr;
	InOther.View_          = nullptr;
	InOther.ImGuiHandle_   = nullptr;
}

OaUiImage& OaUiImage::operator=(OaUiImage&& InOther) noexcept {
	if (this != &InOther) {
		// Destructor on this would deferred-free our binding; for now just
		// drop the pointers since impl is a no-op.
		Tex_                   = std::move(InOther.Tex_);
		DescriptorSet_         = InOther.DescriptorSet_;
		Sampler_               = InOther.Sampler_;
		View_                  = InOther.View_;
		ImGuiHandle_           = InOther.ImGuiHandle_;
		InOther.DescriptorSet_ = nullptr;
		InOther.Sampler_       = nullptr;
		InOther.View_          = nullptr;
		InOther.ImGuiHandle_   = nullptr;
	}
	return *this;
}

OaUiImage::~OaUiImage() {
	// TODO(Step 3b.5): deferred-free DescriptorSet_ / Sampler_ / View_
	// against the engine's ImGui pool at end of current frame.
}
