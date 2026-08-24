// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include "../runtime/engine/borrowedServiceRetirement.h"
#include "../runtime/textureAccess.h"

#include <oa/ui/image.h>
#include <oa/core/log.h>

#include "../../../thirdparty/stb/stb_image.h"

// ─── helpers ──────────────────────────────────────────────────────────────────

static oa::U64 imagePlaneScalarBytes(oa::ImageDtype inDtype) noexcept {
	switch (inDtype) {
		case oa::ImageDtype::U8: return 1U;
		case oa::ImageDtype::U16:
		case oa::ImageDtype::BF16: return 2U;
		case oa::ImageDtype::F32: return 4U;
	}
	return 0U;
}

static oa::Result<oavk::Buffer> uploadImagePlane(
	oa::Engine& inEngine,
	const void* inData,
	oa::U64 inBytes)
{
	if (inData == nullptr or inBytes == 0U) {
		return oa::Status::invalidArgument(
			"oa::ImagePlanes requires non-empty plane data");
	}
	if (inBytes > UINT64_MAX - 3U) {
		return oa::Status::invalidArgument(
			"oa::ImagePlanes plane byte size overflows alignment");
	}
	const oa::U64 paddedBytes = (inBytes + 3U) & ~oa::U64{3U};
	oa::Vec<oa::U8> padded(paddedBytes);
	oa::memcpy(padded.data(), inData, inBytes);
	if (paddedBytes > inBytes) {
		oa::memset(padded.data() + inBytes, 0, paddedBytes - inBytes);
	}
	return oa::TextureAccess::uploadBuffer(
		inEngine, padded.data(), paddedBytes);
}


// ─── oa::ImagePlanes ────────────────────────────────────────────────────────────

oa::ImagePlanes::ImagePlanes(oa::ImagePlanes&& inOther) noexcept
	: engine_(inOther.engine_)
	, planes_(inOther.planes_)
	, dtypes_(inOther.dtypes_)
	, consumerCompletions_(oa::move(inOther.consumerCompletions_))
	, width_(inOther.width_)
	, height_(inOther.height_)
	, channelCount_(inOther.channelCount_)
{
	inOther.engine_ = nullptr;
	inOther.planes_ = {};
	inOther.dtypes_ = {};
	inOther.width_ = 0;
	inOther.height_ = 0;
	inOther.channelCount_ = 0;
}

oa::ImagePlanes& oa::ImagePlanes::operator=(oa::ImagePlanes&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		engine_ = inOther.engine_;
		planes_ = inOther.planes_;
		dtypes_ = inOther.dtypes_;
		consumerCompletions_ = oa::move(inOther.consumerCompletions_);
		width_ = inOther.width_;
		height_ = inOther.height_;
		channelCount_ = inOther.channelCount_;
		inOther.engine_ = nullptr;
		inOther.planes_ = {};
		inOther.dtypes_ = {};
		inOther.width_ = 0;
		inOther.height_ = 0;
		inOther.channelCount_ = 0;
	}
	return *this;
}

oa::ImagePlanes::~ImagePlanes() { abandon_(); }

oa::Result<oa::ImagePlanes> oa::ImagePlanes::loadFile(oa::Engine& inRt, oa::StringView inPath) {
	int w = 0, h = 0, ch = 0;
	// query channel count first
	stbi_info(inPath.data(), &w, &h, &ch);
	if (w == 0 || h == 0) {
		// Try loading to get dimensions
		ch = 4;
	}

	// Determine if HDR
	bool isHdr = stbi_is_hdr(inPath.data()) != 0;
	oa::ImagePlanes planes;
	planes.engine_ = &inRt;

	if (isHdr) {
		float* px = stbi_loadf(inPath.data(), &w, &h, &ch, 0);
		if (!px) {
			OaLogError(oa::LogComponent::Ui, "oa::ImagePlanes::loadFile: HDR load failed: %s", inPath.data());
			return oa::Status::error("stb_image HDR load failed");
		}
		oa::U32 nCh = static_cast<oa::U32>(ch);
		const oa::U64 pixelCount = static_cast<oa::U64>(w)
			* static_cast<oa::U64>(h);
		oa::U64 planeBytes = pixelCount * sizeof(float);

		oa::Vec<float> tmp(pixelCount);
		for (oa::U32 c = 0; c < nCh and c < kImageMaxPlanes; ++c) {
			for (oa::I64 i = 0; i < static_cast<oa::I64>(w) * h; ++i) {
				tmp[static_cast<oa::U64>(i)] = px[static_cast<oa::U64>(i) * nCh + c];
			}
			auto res = uploadImagePlane(inRt, tmp.data(), planeBytes);
			if (!res) { stbi_image_free(px); return res.getStatus(); }
			planes.planes_[c] = oa::move(*res);
			planes.dtypes_[c] = oa::ImageDtype::F32;
		}
		stbi_image_free(px);
		planes.width_        = w;
		planes.height_       = h;
		planes.channelCount_ = static_cast<oa::U8>(std::min(nCh, kImageMaxPlanes));
	} else {
		stbi_uc* px = stbi_load(inPath.data(), &w, &h, &ch, 0);
		if (!px) {
			OaLogError(oa::LogComponent::Ui, "oa::ImagePlanes::loadFile: load failed: %s", inPath.data());
			return oa::Status::error("stb_image load failed");
		}
		oa::U32 nCh = static_cast<oa::U32>(ch);
		const oa::U64 planeBytes = static_cast<oa::U64>(w)
			* static_cast<oa::U64>(h);

		oa::Vec<oa::U8> tmp(planeBytes);
		for (oa::U32 c = 0; c < nCh and c < kImageMaxPlanes; ++c) {
			for (oa::I64 i = 0; i < static_cast<oa::I64>(w) * h; ++i) {
				tmp[static_cast<oa::U64>(i)] = px[static_cast<oa::U64>(i) * nCh + c];
			}
			auto res = uploadImagePlane(inRt, tmp.data(), planeBytes);
			if (!res) { stbi_image_free(px); return res.getStatus(); }
			planes.planes_[c] = oa::move(*res);
			planes.dtypes_[c] = oa::ImageDtype::U8;
		}
		stbi_image_free(px);
		planes.width_        = w;
		planes.height_       = h;
		planes.channelCount_ = static_cast<oa::U8>(std::min(nCh, kImageMaxPlanes));
	}

	OaLogInfo(oa::LogComponent::Ui, "oa::ImagePlanes: loaded %s (%dx%d, %u ch)",
		inPath.data(), w, h, planes.channelCount_);
	return planes;
}

oa::Result<oa::ImagePlanes> oa::ImagePlanes::fromPlanes(
	oa::Engine& inRt,
	oa::Span<const oa::Span<const oa::U8>> inPlanes,
	oa::Span<const oa::ImageDtype>       inDtypes,
	oa::I32 inW,
	oa::I32 inH)
{
	if (inPlanes.size() == 0 or inPlanes.size() > kImageMaxPlanes) {
		return oa::Status::error("FromPlanes: channel count must be 1-4");
	}
	if (inPlanes.size() != inDtypes.size()) {
		return oa::Status::error("FromPlanes: planes/dtypes size mismatch");
	}
	if (inW <= 0 or inH <= 0) {
		return oa::Status::invalidArgument(
			"FromPlanes: dimensions must be positive");
	}
	const oa::U64 width = static_cast<oa::U64>(inW);
	const oa::U64 height = static_cast<oa::U64>(inH);
	if (width > UINT64_MAX / height) {
		return oa::Status::invalidArgument(
			"FromPlanes: pixel count overflows");
	}
	const oa::U64 pixels = width * height;

	oa::ImagePlanes planes;
	planes.engine_ = &inRt;
	planes.width_ = inW;
	planes.height_ = inH;
	planes.channelCount_ = static_cast<oa::U8>(inPlanes.size());

	for (oa::U32 c = 0; c < inPlanes.size(); ++c) {
		const oa::U64 scalarBytes = imagePlaneScalarBytes(inDtypes[c]);
		if (scalarBytes == 0U or pixels > UINT64_MAX / scalarBytes) {
			return oa::Status::invalidArgument(
				"FromPlanes: plane byte size overflows");
		}
		const oa::U64 expectedBytes = pixels * scalarBytes;
		if (inPlanes[c].size() != expectedBytes) {
			return oa::Status::invalidArgument(
				"FromPlanes: plane byte size does not match dimensions and dtype");
		}
		auto res = uploadImagePlane(
			inRt, inPlanes[c].data(), expectedBytes);
		if (!res) return res.getStatus();
		planes.planes_[c] = oa::move(*res);
		planes.dtypes_[c] = inDtypes[c];
	}
	return planes;
}

oa::Status oa::ImagePlanes::markConsumed(const oa::Event& inCompletion) {
	if (not isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::ImagePlanes::markConsumed requires a valid image");
	}
	if (not engine_->ownsEvent(inCompletion)) {
		return oa::Status::invalidArgument(
			"oa::ImagePlanes::markConsumed requires an event from its engine");
	}
	for (oa::Usize index = consumerCompletions_.size(); index > 0U; --index) {
		if (consumerCompletions_[index - 1U].isComplete()) {
			consumerCompletions_.erase(
				consumerCompletions_.begin() + (index - 1U));
		}
	}
	if (inCompletion.isComplete()) return oa::Status::ok();
	for (const oa::Event& completion : consumerCompletions_) {
		if (completion.isSameCompletion(inCompletion)) return oa::Status::ok();
	}
	consumerCompletions_.pushBack(inCompletion);
	return oa::Status::ok();
}

void oa::ImagePlanes::release_() noexcept {
	if (engine_ != nullptr) {
		for (oavk::Buffer& plane : planes_) {
			if (plane.buffer != nullptr) {
				oa::EngineResourceAccess::freeBuffer(*engine_, plane);
			}
		}
	}
	engine_ = nullptr;
	planes_ = {};
	dtypes_ = {};
	consumerCompletions_.clear();
	width_ = 0;
	height_ = 0;
	channelCount_ = 0;
}

void oa::ImagePlanes::abandon_() noexcept {
	if (engine_ == nullptr) return;
	bool hasPendingConsumer = false;
	for (const oa::Event& completion : consumerCompletions_) {
		if (not completion.isComplete()) {
			hasPendingConsumer = true;
			break;
		}
	}
	if (not hasPendingConsumer) {
		release_();
		return;
	}
	oa::Engine* engine = engine_;
	auto retired = oa::makeUnique<oa::ImagePlanes>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::ImagePlanes::completeRetired_,
		&oa::ImagePlanes::releaseRetired_);
}

oa::Status oa::ImagePlanes::completeRetired_(void* inPayload) {
	auto* planes = static_cast<oa::ImagePlanes*>(inPayload);
	if (planes == nullptr) return oa::Status::ok();
	for (const oa::Event& completion : planes->consumerCompletions_) {
		OA_RETURN_IF_ERROR(completion.wait());
	}
	planes->release_();
	return oa::Status::ok();
}

void oa::ImagePlanes::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::ImagePlanes> planes(
		static_cast<oa::ImagePlanes*>(inPayload));
}
