// Engine first — VK_NO_PROTOTYPES before any vulkan header pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/engineAccess.h>
#include <oa/runtime/engine/resourceAccess.h>

#include "textureAccess.h"

#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/matrixAccess.h>
#include <oa/runtime/executionSession.h>

#include <oa/core/memory.h>

oa::U32 oa::Texture::bindlessIndex() const noexcept {
	return bufferOwner_ ? bufferOwner_->bindlessIndex : UINT32_MAX;
}

bool oa::Texture::isImageBacked() const noexcept {
	return image_ != 0U;
}

bool oa::Texture::isValid() const noexcept {
	if (engine_ == nullptr or width_ <= 0 or height_ <= 0) return false;
	if (image_ != 0U) {
		return view_ != 0U and format_ != 0 and layout_ != 0;
	}
	return bufferOwner_ and bufferOwner_->buffer != nullptr
		and bufferOwner_->bindlessIndex != UINT32_MAX;
}

oa::Result<oavk::Buffer> oa::TextureAccess::uploadBuffer(
	oa::Engine& inEngine,
	const void* inData,
	oa::U64 inBytes)
{
	if (inData == nullptr or inBytes == 0U) {
		return oa::Status::invalidArgument(
			"oa::TextureAccess::uploadBuffer requires non-empty data");
	}
	auto stagingResult = oa::EngineResourceAccess::allocBuffer(inEngine, inBytes);
	if (not stagingResult) return stagingResult.getStatus();
	oavk::Buffer staging = oa::move(*stagingResult);
	oa::memcpy(staging.mappedPtr, inData, inBytes);
	if (not oa::EngineAllocatorAccess::get(inEngine).flushHostBuffer(
		staging, 0U, inBytes))
	{
		oa::EngineResourceAccess::freeBuffer(inEngine, staging);
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"oa::TextureAccess::uploadBuffer: staging flush failed");
	}

	auto deviceResult =
		oa::EngineResourceAccess::allocBufferDevice(inEngine, inBytes);
	if (not deviceResult) {
		oa::EngineResourceAccess::freeBuffer(inEngine, staging);
		return deviceResult.getStatus();
	}
	oavk::Buffer device = oa::move(*deviceResult);
	auto copy = oa::EngineResourceAccess::copyBufferAsync(
		inEngine, staging, device, inBytes);
	if (not copy.isOk()) {
		oa::EngineResourceAccess::freeBuffer(inEngine, staging);
		oa::EngineResourceAccess::freeBuffer(inEngine, device);
		return copy.getStatus();
	}
	if (const oa::Status status = copy->wait(); not status.isOk()) {
		oa::EngineResourceAccess::freeBuffer(inEngine, staging);
		oa::EngineResourceAccess::freeBuffer(inEngine, device);
		return status;
	}
	oa::EngineResourceAccess::freeBuffer(inEngine, staging);
	return device;
}

namespace oa {

namespace FnTexture {

oa::Result<oa::Texture> fromPixels(
	oa::Engine& inEngine,
	oa::Span<const oa::U8> inRgba,
	oa::I32 inWidth,
	oa::I32 inHeight)
{
	if (inWidth <= 0 or inHeight <= 0) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromPixels: dimensions must be positive");
	}
	const oa::U64 width = static_cast<oa::U64>(inWidth);
	const oa::U64 height = static_cast<oa::U64>(inHeight);
	if (width > UINT64_MAX / height or width * height > UINT64_MAX / 4U) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromPixels: byte size overflows");
	}
	const oa::U64 expected = width * height * 4U;
	if (inRgba.size() != expected) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromPixels: expected exactly width * height * 4 bytes");
	}
	auto uploaded = oa::TextureAccess::uploadBuffer(
		inEngine, inRgba.data(), expected);
	if (not uploaded) return uploaded.getStatus();
	auto owner = oa::EngineAccess(inEngine).adoptBufferLease(
		oa::move(*uploaded));
	if (not owner) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::FnTexture::fromPixels: failed to adopt buffer ownership");
	}
	return oa::TextureAccess::fromBuffer(
		inEngine, oa::move(owner), inWidth, inHeight);
}

oa::Status copyToHost(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	void* outHost,
	oa::U64 inBytes)
{
	if (outHost == nullptr) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::copyToHost: output pointer is null");
	}
	const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTexture);
	if (not inTexture.isValid() or inTexture.isImageBacked()
		or buffer == nullptr)
	{
		return oa::Status::invalidArgument(
			"oa::FnTexture::copyToHost: texture must be buffer-backed");
	}
	if (oa::TextureAccess::engine(inTexture) != &inEngine
		or buffer->allocatorIdentity
			!= oa::EngineAllocatorAccess::get(inEngine).allocator) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::copyToHost: texture belongs to another engine");
	}
	const oa::U64 required = static_cast<oa::U64>(inTexture.width())
		* static_cast<oa::U64>(inTexture.height()) * 4U;
	if (inBytes < required or buffer->size < required) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::copyToHost: host or device range is too small");
	}
	OA_RETURN_IF_ERROR(oa::FnMatrix::completeRecordedWork(
		oa::ExecutionSession::forEngine(inEngine)));
	return oa::EngineResourceAccess::readbackBuffer(
		inEngine, *buffer, 0U, outHost, required);
}

static oa::Status blitRecorded(
	oa::ExecutionSession& inContext,
	const oa::BlitDesc& inDesc)
{
	if (not inContext.engine().hasCompute()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::FnTexture::blit: context has no compute queue");
	}
	if (inDesc.src == nullptr or inDesc.dst == nullptr) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::blit: source and destination are required");
	}
	if (not inDesc.src->isValid() or not inDesc.dst->isValid()) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::blit: source and destination must be valid");
	}
	if (inDesc.src->isImageBacked() or inDesc.dst->isImageBacked()) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::blit: image-backed presentation belongs to oa::Presenter");
	}
	oa::Engine& engine = inContext.engine();
	if (oa::TextureAccess::engine(*inDesc.src) != &engine
		or oa::TextureAccess::engine(*inDesc.dst) != &engine) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::blit: textures must belong to the context engine");
	}
	if (inDesc.src->width() != inDesc.dst->width()
		or inDesc.src->height() != inDesc.dst->height()) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"oa::FnTexture::blit: scaled blits are not implemented");
	}
	if (not inDesc.srcRect.isEmpty() or not inDesc.dstRect.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::Unimplemented,
			"oa::FnTexture::blit: rectangle blits are not implemented");
	}

	const oa::U64 pixelCount64 = static_cast<oa::U64>(inDesc.src->width())
		* static_cast<oa::U64>(inDesc.src->height());
	if (pixelCount64 > UINT32_MAX) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::FnTexture::blit: pixel count exceeds the kernel contract");
	}
	const oa::U32 pixelCount = static_cast<oa::U32>(pixelCount64);
	const oavk::Buffer* source = oa::TextureAccess::buffer(*inDesc.src);
	const oavk::Buffer* destination = oa::TextureAccess::buffer(*inDesc.dst);
	if (source == nullptr or destination == nullptr
		or source->size < pixelCount64 * 4U
		or destination->size < pixelCount64 * 4U) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::blit: texture buffer range is too small");
	}
	oavk::Buffer buffers[2] = {*source, *destination};
	oa::SharedPtr<oavk::Buffer> owners[2] = {
		oa::TextureAccess::bufferOwner(*inDesc.src),
		oa::TextureAccess::bufferOwner(*inDesc.dst),
	};
	oa::BufferAccess access[2] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	struct Push { oa::U32 Count; } push{pixelCount};
	constexpr oa::U32 kGroupSize = 256U;
	const oa::U32 groupsX = (pixelCount + kGroupSize - 1U) / kGroupSize;
	inContext.add(
		"Copy",
		oa::Span<oavk::Buffer>(buffers, 2),
		oa::Span<oa::SharedPtr<oavk::Buffer>>(owners, 2),
		oa::Span<oa::BufferAccess>(access, 2),
		&push,
		sizeof(push),
		groupsX);
	return oa::Status::ok();
}

oa::Status blit(const oa::BlitDesc& inDesc) {
	return blitRecorded(oa::ExecutionSession::getActive(), inDesc);
}

static oa::Status clearRecorded(
	oa::ExecutionSession& inContext,
	const oa::Texture& inTarget,
	oa::ClearColor inColor)
{
	if (not inContext.engine().hasCompute()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::FnTexture::clear: context has no compute queue");
	}
	if (not inTarget.isValid()) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::clear: target must be a valid positive-extent texture");
	}
	if (inTarget.isImageBacked()) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::clear: image-backed presentation belongs to oa::Presenter");
	}
	oa::Engine& engine = inContext.engine();
	if (oa::TextureAccess::engine(inTarget) != &engine) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::clear: texture belongs to another engine");
	}
	const oa::U64 pixelCount64 = static_cast<oa::U64>(inTarget.width())
		* static_cast<oa::U64>(inTarget.height());
	if (pixelCount64 > UINT32_MAX) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::FnTexture::clear: pixel count exceeds the kernel contract");
	}
	const oa::U32 pixelCount = static_cast<oa::U32>(pixelCount64);
	auto clamp01 = [](oa::F32 inValue) -> oa::U32 {
		const oa::F32 value = inValue < 0.0F
			? 0.0F
			: (inValue > 1.0F ? 1.0F : inValue);
		return static_cast<oa::U32>(value * 255.0F + 0.5F) & 0xFFU;
	};
	const oa::U32 packed =
		(clamp01(inColor.a) << 24U)
		| (clamp01(inColor.b) << 16U)
		| (clamp01(inColor.g) << 8U)
		| clamp01(inColor.r);
	const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTarget);
	if (buffer == nullptr or buffer->size < pixelCount64 * 4U) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::clear: texture buffer range is too small");
	}
	oavk::Buffer buffers[1] = {*buffer};
	oa::SharedPtr<oavk::Buffer> owners[1] = {
		oa::TextureAccess::bufferOwner(inTarget)};
	oa::BufferAccess access[1] = {oa::BufferAccess::Write};
	struct Push {
		oa::U32 Count;
		oa::U32 Rgba;
	} push{pixelCount, packed};
	constexpr oa::U32 kGroupSize = 256U;
	const oa::U32 groupsX = (pixelCount + kGroupSize - 1U) / kGroupSize;
	inContext.add(
		"ClearRgba8",
		oa::Span<oavk::Buffer>(buffers, 1),
		oa::Span<oa::SharedPtr<oavk::Buffer>>(owners, 1),
		oa::Span<oa::BufferAccess>(access, 1),
		&push,
		sizeof(push),
		groupsX);
	return oa::Status::ok();
}

oa::Status clear(const oa::Texture& inTarget, oa::ClearColor inColor) {
	return clearRecorded(oa::ExecutionSession::getActive(), inTarget, inColor);
}

static oa::Result<oa::Texture> fromImageRecorded(
	oa::ExecutionSession& inContext,
	const oa::Image& inImage)
{
	if (not inContext.engine().hasCompute()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::FnTexture::fromImage: context has no compute queue");
	}
	if (not inImage.validate() or inImage.isEmpty()
		or inImage.batchSize() != 1) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromImage: expected one valid non-empty image");
	}
	if (inImage.layout() == oa::ImageLayout::Hw
		and inImage.format() != oa::ImageFormat::Gray) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromImage: HW layout is gray-only");
	}
	const oa::Matrix& matrix = inImage.asMatrix();
	if (matrix.getDtype() != oa::ScalarType::Float32
		and matrix.getDtype() != oa::ScalarType::BFloat16) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromImage: expected floating-point storage");
	}
	const oa::I32 channels = inImage.channels();
	const oa::I32 height = inImage.height();
	const oa::I32 width = inImage.width();
	if (channels < 1 or channels > 4 or width <= 0 or height <= 0) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromImage: invalid channels or extent");
	}
	const oa::U64 pixelCount64 = static_cast<oa::U64>(width)
		* static_cast<oa::U64>(height);
	if (pixelCount64 > UINT32_MAX) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"oa::FnTexture::fromImage: pixel count exceeds the kernel contract");
	}
	const oavk::Buffer input = oa::MatrixAccess::descriptor(matrix);
	oa::Engine& engine = inContext.engine();
	if (input.buffer == nullptr or input.allocatorIdentity == nullptr
		or input.allocatorIdentity
			!= oa::EngineAllocatorAccess::get(engine).allocator) {
		return oa::Status::invalidArgument(
			"oa::FnTexture::fromImage: image storage belongs to another engine");
	}
	oa::ExecutionSession::RecordingScope recording(inContext);
	// UInt32 is exactly one packed RGBA8 pixel. It also gives the graph a shared
	// allocation owner, avoiding a raw-buffer lifetime escape. Image layout and
	// channel format remain explicit rather than inferred from a matrix.
	auto packed = oa::FnMatrix::empty(
		oa::MatrixShape{static_cast<oa::I64>(pixelCount64)}, oa::ScalarType::UInt32);
	if (not packed.hasStorage()) {
		return oa::Status::error(
			oa::StatusCode::ResourceExhausted,
			"oa::FnTexture::fromImage: packed texture allocation failed");
	}
	struct Push {
		oa::U32 channels;
		oa::U32 height;
		oa::U32 width;
		oa::U32 layout;
		oa::U32 format;
	} push{
		static_cast<oa::U32>(channels),
		static_cast<oa::U32>(height),
		static_cast<oa::U32>(width),
		static_cast<oa::U32>(inImage.layout()),
		static_cast<oa::U32>(inImage.format()),
	};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,
		oa::BufferAccess::Write,
	};
	inContext.add(
		"MatrixToRgba8",
		{&matrix, &packed},
		access,
		&push,
		sizeof(push),
		(static_cast<oa::U32>(pixelCount64) + 255U) / 256U);
	return oa::TextureAccess::fromBuffer(
		engine, oa::MatrixAccess::storageOwner(packed), width, height);
}

oa::Result<oa::Texture> fromImage(
	oa::Engine& inEngine,
	const oa::Image& inImage)
{
	return fromImageRecorded(
		oa::ExecutionSession::forEngine(inEngine), inImage);
}

} // namespace FnTexture

} // namespace oa
