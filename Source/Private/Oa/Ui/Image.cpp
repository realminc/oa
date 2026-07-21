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

static OaContext& ContextForEngine(OaEngine& InEngine) {
	OaContext* active = OaContext::GetDefaultPtr();
	return active != nullptr and active->GetEngine() == &InEngine
		? *active : InEngine.GetContext();
}

static OaResult<OaVkBuffer> UploadToDevice(
	OaEngine& InRt,
	const void* InData,
	OaU64 InBytes)
{
	auto stagingResult = InRt.AllocBuffer(InBytes);
	if (not stagingResult) return stagingResult.GetStatus();
	OaVkBuffer staging = std::move(*stagingResult);
	std::memcpy(staging.MappedPtr, InData, InBytes);
	if (not InRt.Allocator.FlushHostBuffer(staging, 0U, InBytes)) {
		InRt.FreeBuffer(staging);
		return OaStatus::Error(OaStatusCode::VulkanError,
			"UploadToDevice: staging flush failed");
	}

	auto deviceRes = InRt.AllocBufferDevice(InBytes);
	if (not deviceRes) {
		InRt.FreeBuffer(staging);
		return deviceRes.GetStatus();
	}
	OaVkBuffer device = *deviceRes;

	auto copy = InRt.CopyBufferAsync(staging, device, InBytes);
	if (not copy.IsOk()) {
		InRt.FreeBuffer(staging);
		InRt.FreeBuffer(device);
		return copy.GetStatus();
	}
	if (const OaStatus status = copy->Wait(); not status.IsOk()) {
		InRt.FreeBuffer(staging);
		InRt.FreeBuffer(device);
		return status;
	}

	InRt.FreeBuffer(staging);
	return device;
}


// ─── OaTexture ────────────────────────────────────────────────────────────────

OaResult<OaTexture> OaTexture::LoadFile(OaEngine& InRt, OaStringView InPath) {
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
	OaEngine& InRt,
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

void OaTexture::Destroy(OaEngine& InRt) {
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


// ─── OaFnTexture ─────────────────────────────────────────────────────────────

namespace OaFnTexture {

OaStatus Blit(OaContext& InContext, const OaBlitDesc& InDesc) {
	if (not InContext.HasCompute()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaFnTexture::Blit: context has no compute queue");
	}
	if (InDesc.Src == nullptr or InDesc.Dst == nullptr) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Blit: source and destination are required");
	}
	if (not InDesc.Src->IsValid() or not InDesc.Dst->IsValid()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Blit: source and destination must be valid");
	}
	if (InDesc.Src->IsImageBacked() or InDesc.Dst->IsImageBacked()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Blit: image-backed presentation belongs to OaPresenter");
	}
	if (InDesc.Src->Width <= 0 or InDesc.Src->Height <= 0
		or InDesc.Dst->Width <= 0 or InDesc.Dst->Height <= 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Blit: texture extents must be positive");
	}
	if (InDesc.Src->Width != InDesc.Dst->Width
		or InDesc.Src->Height != InDesc.Dst->Height) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"OaFnTexture::Blit: scaled blits are not implemented");
	}
	if (not InDesc.SrcRect.IsEmpty() or not InDesc.DstRect.IsEmpty()) {
		return OaStatus::Error(OaStatusCode::Unimplemented,
			"OaFnTexture::Blit: rectangle blits are not implemented");
	}

	const OaU64 pixelCount64 = static_cast<OaU64>(InDesc.Src->Width)
		* static_cast<OaU64>(InDesc.Src->Height);
	if (pixelCount64 > 0xFFFFFFFFULL) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"OaFnTexture::Blit: pixel count exceeds the kernel contract");
	}
	const OaU32 pixelCount = static_cast<OaU32>(pixelCount64);
	OaVkBuffer buffers[2] = {
		InDesc.Src->DeviceBuf,
		InDesc.Dst->DeviceBuf,
	};
	OaBufferAccess access[2] = {
		OaBufferAccess::Read,
		OaBufferAccess::Write,
	};
	struct Push { OaU32 Count; } push{pixelCount};
	constexpr OaU32 kGroupSize = 256U;
	const OaU32 groupsX = (pixelCount + kGroupSize - 1U) / kGroupSize;
	InContext.Add(
		"Copy",
		OaSpan<OaVkBuffer>(buffers, 2),
		OaSpan<OaBufferAccess>(access, 2),
		&push,
		sizeof(push),
		groupsX);
	return OaStatus::Ok();
}

OaStatus Blit(const OaBlitDesc& InDesc) {
	return Blit(OaContext::GetDefault(), InDesc);
}

OaStatus Clear(
	OaContext& InContext,
	const OaTexture& InTarget,
	OaClearColor InColor)
{
	if (not InContext.HasCompute()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaFnTexture::Clear: context has no compute queue");
	}
	if (not InTarget.IsValid() or InTarget.Width <= 0 or InTarget.Height <= 0) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Clear: target must be a valid positive-extent texture");
	}
	if (InTarget.IsImageBacked()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaFnTexture::Clear: image-backed presentation belongs to OaPresenter");
	}

	const OaU64 pixelCount64 = static_cast<OaU64>(InTarget.Width)
		* static_cast<OaU64>(InTarget.Height);
	if (pixelCount64 > 0xFFFFFFFFULL) {
		return OaStatus::Error(OaStatusCode::OutOfRange,
			"OaFnTexture::Clear: pixel count exceeds the kernel contract");
	}
	const OaU32 pixelCount = static_cast<OaU32>(pixelCount64);
	auto clamp01 = [](OaF32 InValue) -> OaU32 {
		const OaF32 value = InValue < 0.0F
			? 0.0F
			: (InValue > 1.0F ? 1.0F : InValue);
		return static_cast<OaU32>(value * 255.0F + 0.5F) & 0xFFU;
	};
	const OaU32 packed =
		(clamp01(InColor.A) << 24U)
		| (clamp01(InColor.B) << 16U)
		| (clamp01(InColor.G) << 8U)
		| clamp01(InColor.R);
	OaVkBuffer buffers[1] = {InTarget.DeviceBuf};
	OaBufferAccess access[1] = {OaBufferAccess::Write};
	struct Push {
		OaU32 Count;
		OaU32 Rgba;
	} push{pixelCount, packed};
	constexpr OaU32 kGroupSize = 256U;
	const OaU32 groupsX = (pixelCount + kGroupSize - 1U) / kGroupSize;
	InContext.Add(
		"ClearRgba8",
		OaSpan<OaVkBuffer>(buffers, 1),
		OaSpan<OaBufferAccess>(access, 1),
		&push,
		sizeof(push),
		groupsX);
	return OaStatus::Ok();
}

OaStatus Clear(const OaTexture& InTarget, OaClearColor InColor) {
	return Clear(OaContext::GetDefault(), InTarget, InColor);
}

} // namespace OaFnTexture


// ─── OaImagePlanes ────────────────────────────────────────────────────────────

OaResult<OaImagePlanes> OaImagePlanes::LoadFile(OaEngine& InRt, OaStringView InPath) {
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
	OaEngine& InRt,
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
	OaContext& InContext,
	const OaMatrix& InMatrix)
{
	if (not InContext.HasCompute()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaTexture::FromMatrix: context has no compute queue");
	}
	const auto shape = InMatrix.GetShape();
	if (shape.Rank != 4 or shape[0] != 1) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: expected [1,C,H,W] matrix");
	}
	const OaI32 channels = static_cast<OaI32>(shape[1]);
	if (channels != 1 and channels != 3 and channels != 4) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: channel count must be 1, 3 or 4");
	}
	if (InMatrix.GetDtype() != OaScalarType::Float32
		and InMatrix.GetDtype() != OaScalarType::BFloat16
		and InMatrix.GetDtype() != OaScalarType::Float16) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: expected a floating-point matrix");
	}
	const OaI32 h = static_cast<OaI32>(shape[2]);
	const OaI32 w = static_cast<OaI32>(shape[3]);
	if (w <= 0 || h <= 0) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: dimensions must be positive");
	}
	const OaVkBuffer input = InMatrix.GetVkBuffer();
	OaEngine& engine = InContext.Engine();
	if (input.Buffer == nullptr or input.AllocatorIdentity == nullptr
		or input.NodeIndex != 0U
		or input.AllocatorIdentity != engine.Allocator.Allocator) {
		return OaStatus::InvalidArgument(
			"OaTexture::FromMatrix: matrix storage must belong to the context engine");
	}

	OaContext::RecordingScope recording(InContext);
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
	InContext.Add("MatrixToRgba8", {&InMatrix, &packed}, access,
		&push, sizeof(push),
		(static_cast<OaU32>(pixelCount) + 255U) / 256U);

	OaTexture texture;
	texture.DeviceBuf = packed.GetVkBuffer();
	texture.DeviceOwner = packed.VkBuf_;
	texture.Width = w;
	texture.Height = h;
	return texture;
}

OaResult<OaTexture> OaTexture::FromMatrix(
	OaEngine& InRt,
	const OaMatrix& InMatrix)
{
	return FromMatrix(ContextForEngine(InRt), InMatrix);
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
	OaContext& InContext,
	const OaImage& InImage)
{
	return FromMatrix(InContext, InImage.AsMatrix());
}

OaResult<OaTexture> OaTexture::FromImage(
	OaEngine& InRt,
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
	OaEngine& InRt,
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

void OaImagePlanes::Destroy(OaEngine& InRt) {
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
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() - InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator*(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() * InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator/(const OaTexture& InOther) const {
	OaMatrix result = this->ToMatrix() / InOther.ToMatrix();
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator+(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() + InScalar;
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() - InScalar;
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator*(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() * InScalar;
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator/(OaF32 InScalar) const {
	OaMatrix result = this->ToMatrix() / InScalar;
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
	return img.IsOk() ? *img : OaTexture{};
}

OaTexture OaTexture::operator-() const {
	OaMatrix result = -this->ToMatrix();
	auto img = OaTexture::FromMatrix(OaContext::GetDefault(), result);
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
//     descriptor set against ImGuiPool_ on OaPresenter
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
