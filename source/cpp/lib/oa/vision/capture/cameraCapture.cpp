// OA Vision — oa::CameraCapture implementation (SDL3 camera -> GPU ring buffer)
//
// SDL3 camera API:
//   SDL_GetCameras         — enumerate device IDs
//   SDL_OpenCamera         — open with requested spec
//   SDL_AcquireCameraFrame — returns SDL_Surface* (any format)
//   SDL_ReleaseCameraFrame — must be released before next acquire
//   SDL_ConvertSurface     — pixel-format conversion to RGBA8
//
// ring buffer strategy:
//   N host-visible oavk::Buffer objects, each w*h*4 bytes.
//   Each buffer registered in the bindless heap so bindlessIndex
//   can be passed directly to the UI image path — zero extra copy.
//   On poll(): acquire surface → blit/convert to ring[head] → advance head.

#include <oa/vision/cameraCapture.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/format.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/core/std/utility.h>
#include <oa/runtime/externalMemory.h>
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_camera.h>

#include <stdio.h>

#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <libdrm/drm_fourcc.h>
#endif

// ─── oa::CameraCapture::Impl ───────────────────────────────────────────────────

struct oa::CameraCapture::Impl {
	oa::Engine* rt        = nullptr;
	SDL_Camera*        camera    = nullptr;
	SDL_CameraSpec     spec      = {};

	oa::I32  w           = 0;
	oa::I32  h           = 0;
	oa::I32  fps         = 0;
	oa::I32  ringN       = 0;
	oa::I32  head        = 0;   // next buffer to write
	oa::I32  latest      = -1;  // last written buffer (-1 = no frame yet)
	bool   streaming   = false;

	oa::Vector<oavk::Buffer> ring;
	oa::Vector<oa::Event> ringConsumers;
	oa::CameraCaptureConfig config = {};
	oa::U64 formatGen = 0U;
	oa::U64 reconnects = 0U;

#if defined(__linux__)
	struct V4l2Slot {
		oa::ImportedDmaBufImage imported;
		int exportFd = -1;
		oa::U32 index = 0U;
		bool dequeued = false;
		bool releaseRequested = false;
		oa::Event pendingConsumer;
	};
	int v4l2Fd = -1;
	oa::Vector<V4l2Slot> v4l2Slots;
	VkFormat v4l2VkFormat = VK_FORMAT_UNDEFINED;
	bool v4l2Streaming = false;
	bool v4l2ReconnectEnabled = false;
	oa::U32 consecutiveFailures = 0U;
	oa::SteadyTimePoint nextReconnect = {};

	bool initV4l2();
	[[nodiscard]] oa::Status destroyV4l2(bool inWaitConsumers = true);
	bool pollV4l2(oa::VideoFrame& outFrame, oa::U64& outTimestampUs);
	void requeueCompletedV4l2();
	void releaseV4l2(const oa::VideoFrame& inFrame, const oa::Event& inConsumed);
#endif

	void freeRing() {
		for (auto& b : ring) {
			if (b.buffer) {
				oa::EngineBindlessAccess::deregisterBuffer(*rt, b);
				oa::EngineResourceAccess::freeBuffer(*rt, b);
			}
		}
		ring.clear();
		ringConsumers.clear();
	}

	[[nodiscard]] oa::Status waitRingConsumers() const {
		oa::Status firstError = oa::Status::ok();
		for (const auto& consumer : ringConsumers) {
			const auto status = consumer.wait();
			if (firstError.isOk() and not status.isOk()) firstError = status;
		}
		return firstError;
	}

	void releaseRing(
		const oa::VideoFrame& inFrame,
		const oa::Event& inConsumed)
	{
		if (inFrame.buffer == nullptr) return;
		for (oa::Usize index = 0U; index < ring.size(); ++index) {
			if (inFrame.buffer != &ring[index]) continue;
			ringConsumers[index] = inConsumed;
			return;
		}
	}
};

#if defined(__linux__)
namespace {

int v4l2Ioctl(int inFd, unsigned long inRequest, void* inArgument)
{
	int result;
	do { result = ::ioctl(inFd, inRequest, inArgument); }
	while (result < 0 and errno == EINTR);
	return result;
}

struct CameraPackedFormat {
	oa::U32 v4l2 = 0U;
	VkFormat vulkan = VK_FORMAT_UNDEFINED;
};

constexpr CameraPackedFormat kCameraFormats[] = {
	// On little-endian Linux these V4L2 layouts are byte-compatible with the
	// matching vulkan UNORM formats. X/alpha is ignored by RGBA->nV12.
	{V4L2_PIX_FMT_XBGR32, VK_FORMAT_R8G8B8A8_UNORM},
	{V4L2_PIX_FMT_ABGR32, VK_FORMAT_R8G8B8A8_UNORM},
	{V4L2_PIX_FMT_XRGB32, VK_FORMAT_B8G8R8A8_UNORM},
	{V4L2_PIX_FMT_ARGB32, VK_FORMAT_B8G8R8A8_UNORM},
};

} // namespace

bool oa::CameraCapture::Impl::initV4l2()
{
	(void)destroyV4l2(true);
	oa::String path = config.devicePath;
	if (path.empty()) {
		path = oa::String("/dev/video")
			+ oa::toString(static_cast<oa::I64>(config.deviceIndex));
	}
	v4l2Fd = ::open(path.cStr(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
	if (v4l2Fd < 0) return false;

	v4l2_capability capabilities = {};
	if (v4l2Ioctl(v4l2Fd, VIDIOC_QUERYCAP, &capabilities) < 0
		or (capabilities.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0U
		or (capabilities.device_caps & V4L2_CAP_STREAMING) == 0U) {
		(void)destroyV4l2(false);
		return false;
	}

	v4l2_format selected = {};
	selected.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bool found = false;
	for (const auto& candidate : kCameraFormats) {
		v4l2_format format = {};
		format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		format.fmt.pix.width = static_cast<oa::U32>(config.width);
		format.fmt.pix.height = static_cast<oa::U32>(config.height);
		format.fmt.pix.pixelformat = candidate.v4l2;
		format.fmt.pix.field = V4L2_FIELD_ANY;
		if (v4l2Ioctl(v4l2Fd, VIDIOC_S_FMT, &format) == 0
			and format.fmt.pix.pixelformat == candidate.v4l2) {
			selected = format;
			v4l2VkFormat = candidate.vulkan;
			found = true;
			break;
		}
	}
	if (not found) { (void)destroyV4l2(false); return false; }

	v4l2_streamparm parameters = {};
	parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	parameters.parm.capture.timeperframe.numerator = 1U;
	parameters.parm.capture.timeperframe.denominator = static_cast<oa::U32>(config.fps);
	(void)v4l2Ioctl(v4l2Fd, VIDIOC_S_PARM, &parameters);
	v4l2_event_subscription subscription = {};
	subscription.type = V4L2_EVENT_SOURCE_CHANGE;
	(void)v4l2Ioctl(v4l2Fd, VIDIOC_SUBSCRIBE_EVENT, &subscription);

	v4l2_requestbuffers request = {};
	request.count = static_cast<oa::U32>(oa::max(2, config.ringFrames));
	request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	request.memory = V4L2_MEMORY_MMAP;
	if (v4l2Ioctl(v4l2Fd, VIDIOC_REQBUFS, &request) < 0 or request.count < 2U) {
		(void)destroyV4l2(false);
		return false;
	}

	v4l2Slots.resize(request.count);
	for (oa::U32 index = 0U; index < request.count; ++index) {
		v4l2_buffer buffer = {};
		buffer.type = request.type;
		buffer.memory = request.memory;
		buffer.index = index;
		if (v4l2Ioctl(v4l2Fd, VIDIOC_QUERYBUF, &buffer) < 0) {
			(void)destroyV4l2(false); return false;
		}
		v4l2_exportbuffer exportBuffer = {};
		exportBuffer.type = request.type;
		exportBuffer.index = index;
		exportBuffer.flags = O_CLOEXEC;
		if (v4l2Ioctl(v4l2Fd, VIDIOC_EXPBUF, &exportBuffer) < 0) {
			(void)destroyV4l2(false); return false;
		}
		auto& slot = v4l2Slots[index];
		slot.exportFd = exportBuffer.fd;
		slot.index = index;
		oa::DmaBufImageDesc description;
		description.fd = slot.exportFd;
		description.width = selected.fmt.pix.width;
		description.height = selected.fmt.pix.height;
		description.format = v4l2VkFormat;
		description.modifier = DRM_FORMAT_MOD_LINEAR;
		description.rowPitch = selected.fmt.pix.bytesperline;
		auto imported = oa::ImportedDmaBufImage::import(*rt, description);
		if (not imported.isOk()) {
			(void)destroyV4l2(false); return false;
		}
		slot.imported = oa::move(*imported);
		if (v4l2Ioctl(v4l2Fd, VIDIOC_QBUF, &buffer) < 0) {
			(void)destroyV4l2(false); return false;
		}
	}
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (v4l2Ioctl(v4l2Fd, VIDIOC_STREAMON, &type) < 0) {
		(void)destroyV4l2(false); return false;
	}
	v4l2Streaming = true;
	v4l2ReconnectEnabled = true;
	w = static_cast<oa::I32>(selected.fmt.pix.width);
	h = static_cast<oa::I32>(selected.fmt.pix.height);
	fps = config.fps;
	if (parameters.parm.capture.timeperframe.numerator > 0U) {
		fps = static_cast<oa::I32>(parameters.parm.capture.timeperframe.denominator
			/ parameters.parm.capture.timeperframe.numerator);
	}
	if (fps <= 0) fps = config.fps;
	streaming = true;
	consecutiveFailures = 0U;
	++formatGen;
	OaLogInfo(oa::LogComponent::Vision,
		"oa::CameraCapture: V4L2 DMA-BUF {} {}x{} @ {} fps, ring={}",
		path.cStr(), w, h, fps, request.count);
	return true;
}

oa::Status oa::CameraCapture::Impl::destroyV4l2(bool inWaitConsumers)
{
	oa::Status firstError = oa::Status::ok();
	if (inWaitConsumers) {
		for (auto& slot : v4l2Slots) {
			if (not slot.releaseRequested) continue;
			const auto status = slot.pendingConsumer.wait();
			if (firstError.isOk() and not status.isOk()) firstError = status;
		}
	}
	if (v4l2Fd >= 0 and v4l2Streaming) {
		v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		(void)v4l2Ioctl(v4l2Fd, VIDIOC_STREAMOFF, &type);
	}
	v4l2Streaming = false;
	for (auto& slot : v4l2Slots) {
		slot.imported = {};
		if (slot.exportFd >= 0) ::close(slot.exportFd);
		slot.exportFd = -1;
	}
	v4l2Slots.clear();
	if (v4l2Fd >= 0) ::close(v4l2Fd);
	v4l2Fd = -1;
	v4l2VkFormat = VK_FORMAT_UNDEFINED;
	return firstError;
}

void oa::CameraCapture::Impl::requeueCompletedV4l2()
{
	for (auto& slot : v4l2Slots) {
		if (not slot.dequeued or not slot.releaseRequested
			or not slot.pendingConsumer.isComplete()) continue;
		v4l2_buffer buffer = {};
		buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = slot.index;
		if (v4l2Ioctl(v4l2Fd, VIDIOC_QBUF, &buffer) == 0) {
			slot.dequeued = false;
			slot.releaseRequested = false;
			slot.pendingConsumer = {};
		}
	}
}

bool oa::CameraCapture::Impl::pollV4l2(oa::VideoFrame& outFrame, oa::U64& outTimestampUs)
{
	requeueCompletedV4l2();
	v4l2_event event = {};
	if (v4l2Ioctl(v4l2Fd, VIDIOC_DQEVENT, &event) == 0
		and event.type == V4L2_EVENT_SOURCE_CHANGE) {
		v4l2Streaming = false;
		streaming = false;
		return false;
	}
	v4l2_buffer buffer = {};
	buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buffer.memory = V4L2_MEMORY_MMAP;
	if (v4l2Ioctl(v4l2Fd, VIDIOC_DQBUF, &buffer) < 0) {
		if (errno == EAGAIN) return false;
		++consecutiveFailures;
		v4l2Streaming = false;
		streaming = false;
		return false;
	}
	if (buffer.index >= v4l2Slots.size()) return false;
	auto& slot = v4l2Slots[buffer.index];
	slot.dequeued = true;
	slot.releaseRequested = false;
	slot.pendingConsumer = {};
	outTimestampUs = static_cast<oa::U64>(buffer.timestamp.tv_sec) * 1'000'000ULL
		+ static_cast<oa::U64>(buffer.timestamp.tv_usec);
	outFrame = {};
	outFrame.resource = oa::VideoFrameResource::Image;
	outFrame.image = slot.imported.image();
	outFrame.imageView = slot.imported.view();
	outFrame.layout = VK_IMAGE_LAYOUT_GENERAL;
	outFrame.externalQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
	outFrame.format = v4l2VkFormat;
	outFrame.width = slot.imported.width();
	outFrame.height = slot.imported.height();
	outFrame.presentationTimestamp = outTimestampUs;
	outFrame.duration = fps > 0 ? 1'000'000ULL / static_cast<oa::U64>(fps) : 0U;
	outFrame.isRgb = true;
	outFrame.colorSpace = oa::YCbCrModel::BT709;
	outFrame.fullRange = true;
	consecutiveFailures = 0U;
	return true;
}

void oa::CameraCapture::Impl::releaseV4l2(
	const oa::VideoFrame& inFrame, const oa::Event& inConsumed)
{
	for (auto& slot : v4l2Slots) {
		if (slot.dequeued and slot.imported.image() == inFrame.image) {
			slot.releaseRequested = true;
			slot.pendingConsumer = inConsumed;
			requeueCompletedV4l2();
			return;
		}
	}
}
#endif

// ─── move / dtor ─────────────────────────────────────────────────────────────

oa::CameraCapture::CameraCapture() = default;

oa::CameraCapture::CameraCapture(oa::CameraCapture&& inOther) noexcept
	: impl_(oa::move(inOther.impl_))
	, width_(inOther.width_)
	, height_(inOther.height_)
	, fps_(inOther.fps_)
	, streaming_(inOther.streaming_)
	, latestTimestampUs_(inOther.latestTimestampUs_)
{
	inOther.width_ = 0;
	inOther.height_ = 0;
	inOther.fps_ = 0;
	inOther.streaming_ = false;
	inOther.latestTimestampUs_ = 0;
}

oa::CameraCapture& oa::CameraCapture::operator=(oa::CameraCapture&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		impl_ = oa::move(inOther.impl_);
		width_ = inOther.width_;
		height_ = inOther.height_;
		fps_ = inOther.fps_;
		streaming_ = inOther.streaming_;
		latestTimestampUs_ = inOther.latestTimestampUs_;
		inOther.width_ = 0;
		inOther.height_ = 0;
		inOther.fps_ = 0;
		inOther.streaming_ = false;
		inOther.latestTimestampUs_ = 0;
	}
	return *this;
}

oa::CameraCapture::~CameraCapture() {
	abandon_();
}

void oa::CameraCapture::abandon_() noexcept {
	if (not impl_) return;
	oa::Engine* engine = impl_->rt;
	if (engine == nullptr) {
		impl_.reset();
		return;
	}
	auto retired = oa::makeUnique<oa::CameraCapture>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::CameraCapture::completeRetired_,
		&oa::CameraCapture::releaseRetired_);
}

oa::Status oa::CameraCapture::completeRetired_(void* inPayload) {
	auto* capture = static_cast<oa::CameraCapture*>(inPayload);
	return capture ? capture->close() : oa::Status::ok();
}

void oa::CameraCapture::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::CameraCapture> capture(
		static_cast<oa::CameraCapture*>(inPayload));
}

// ─── open ────────────────────────────────────────────────────────────────────

oa::Result<oa::CameraCapture> oa::CameraCapture::open(
	oa::Engine& inEngine,
	const oa::CameraCaptureConfig& inConfig)
{
	oa::CameraCapture capture;
	const oa::Status status = capture.init_(inEngine, inConfig);
	if (not status.isOk()) return status;
	return capture;
}

oa::Status oa::CameraCapture::init_(
	oa::Engine& inRt, const oa::CameraCaptureConfig& inConfig) {
	OA_RETURN_IF_ERROR(close());
	if (inConfig.deviceIndex < 0 or inConfig.width <= 0 or inConfig.height <= 0
		or inConfig.fps <= 0 or inConfig.ringFrames < 2) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::CameraCapture requires a non-negative device, positive extent/fps, and ring >= 2");
	}
	impl_ = oa::makeUnique<Impl>();
	impl_->rt   = &inRt;
	impl_->config = inConfig;
	impl_->w    = inConfig.width;
	impl_->h    = inConfig.height;
	impl_->fps  = inConfig.fps;
	impl_->ringN = oa::max(2, inConfig.ringFrames);

#if defined(__linux__)
	if (inConfig.preferDmaBuf and impl_->initV4l2()) {
		width_ = impl_->w;
		height_ = impl_->h;
		fps_ = impl_->fps;
		streaming_ = true;
		return oa::Status::ok();
	}
	if (inConfig.preferDmaBuf) {
		OaLogInfo(oa::LogComponent::Vision,
			"oa::CameraCapture: V4L2 DMA-BUF unavailable; using SDL mapped fallback");
	}
#endif

	if (!SDL_InitSubSystem(SDL_INIT_CAMERA)) {
		OaLogError(oa::LogComponent::Vision, "SDL_INIT_CAMERA failed: {}", SDL_GetError());
		return oa::Status::error("oa::CameraCapture: SDL_INIT_CAMERA failed");
	}

	int numCams = 0;
	SDL_CameraID* camIds = SDL_GetCameras(&numCams);
	if (!camIds || numCams == 0) {
		SDL_free(camIds);
		return oa::Status::error("oa::CameraCapture: no cameras found");
	}

	oa::I32 devIdx = oa::min(inConfig.deviceIndex, numCams - 1);
	SDL_CameraID id = camIds[devIdx];
	SDL_free(camIds);

	SDL_CameraSpec desired = {};
	desired.format     = SDL_PIXELFORMAT_RGBA32;
	desired.width      = inConfig.width;
	desired.height     = inConfig.height;
	desired.framerate_numerator   = inConfig.fps;
	desired.framerate_denominator = 1;

	impl_->camera = SDL_OpenCamera(id, &desired);
	if (!impl_->camera) {
		OaLogError(oa::LogComponent::Vision, "SDL_OpenCamera failed: {}", SDL_GetError());
		return oa::Status::error("oa::CameraCapture: SDL_OpenCamera failed");
	}

	// wait for camera permission grant (SDL3 requires explicit poll loop)
	// camera permission is asynchronous. Keep the deadline finite so a broken
	// backend cannot hang a headless process forever.
	bool permissionResolved = false;
	for (oa::I32 retry = 0; retry < 5000; ++retry) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_EVENT_CAMERA_DEVICE_APPROVED) {
				OaLogInfo(oa::LogComponent::Vision,
					"oa::CameraCapture: camera approved (device {})", devIdx);
				permissionResolved = true;
				break;
			}
			if (e.type == SDL_EVENT_CAMERA_DEVICE_DENIED) {
				SDL_CloseCamera(impl_->camera);
				impl_->camera = nullptr;
				return oa::Status::error("oa::CameraCapture: camera access denied");
			}
		}
		if (permissionResolved) break;
		SDL_Delay(1);
	}
	if (not permissionResolved) {
		const oa::Status permissionFailure = oa::Status::error(
			oa::StatusCode::DeadlineExceeded,
			"oa::CameraCapture: camera permission timed out");
		const oa::Status closeStatus = close();
		if (closeStatus.isOk()) return permissionFailure;
		oa::String message = permissionFailure.toString();
		message += "; cleanup also failed: ";
		message += closeStatus.toString();
		return oa::Status::error(closeStatus.getCode(), oa::move(message));
	}

	// Read actual format after approval
	if (SDL_GetCameraFormat(impl_->camera, &impl_->spec)) {
		impl_->w   = impl_->spec.width;
		impl_->h   = impl_->spec.height;
		if (impl_->spec.framerate_denominator > 0) {
			const oa::I64 numerator = impl_->spec.framerate_numerator;
			const oa::I64 denominator = impl_->spec.framerate_denominator;
			impl_->fps = static_cast<oa::I32>((numerator + denominator / 2) / denominator);
		}
		if (impl_->fps <= 0) impl_->fps = inConfig.fps;
		OaLogInfo(oa::LogComponent::Vision,
			"oa::CameraCapture: opened {}x{} @ {} fps (format=0x{:X})",
			impl_->w, impl_->h, impl_->fps,
			static_cast<oa::U32>(impl_->spec.format));
	}

	// allocate GPU ring buffers (host-visible for direct memcpy)
	oa::U64 frameBytes = static_cast<oa::U64>(impl_->w)
		* static_cast<oa::U64>(impl_->h) * 4ULL;
	impl_->ring.resize(static_cast<oa::Usize>(impl_->ringN));
	impl_->ringConsumers.resize(static_cast<oa::Usize>(impl_->ringN));
	for (oa::I32 i = 0; i < impl_->ringN; ++i) {
		auto res = oa::EngineResourceAccess::allocBuffer(inRt, frameBytes);
		if (!res.isOk()) {
			// Free already-allocated ring entries
			for (oa::I32 j = 0; j < i; ++j) {
				auto& b = impl_->ring[static_cast<size_t>(j)];
				oa::EngineBindlessAccess::deregisterBuffer(inRt, b);
				oa::EngineResourceAccess::freeBuffer(inRt, b);
			}
			OaLogError(oa::LogComponent::Vision, "ring buffer alloc failed: {}",
				res.getStatus().toString().cStr());
			return oa::Status::error("oa::CameraCapture: ring buffer alloc failed");
		}
		impl_->ring[static_cast<oa::Usize>(i)] = oa::move(res.getValue());
		if (oa::EngineBindlessAccess::registerBuffer(
			inRt,
			impl_->ring[static_cast<size_t>(i)]) == OA_BINDLESS_INVALID)
		{
			impl_->freeRing();
			return oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"oa::CameraCapture: bindless ring registration failed");
		}
	}

	width_     = impl_->w;
	height_    = impl_->h;
	fps_       = impl_->fps;
	streaming_ = true;

	OaLogInfo(oa::LogComponent::Vision,
		"oa::CameraCapture: ring x{}, {:.1f} MB/frame",
		impl_->ringN, frameBytes / 1e6f);

	return oa::Status::ok();
}

// ─── Close ─────────────────────────────────────────────────────────────────

oa::Status oa::CameraCapture::close() {
	if (not impl_) return oa::Status::ok();
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() and not inStatus.isOk()) firstError = inStatus;
	};
	streaming_ = false;
#if defined(__linux__)
	retainError(impl_->destroyV4l2());
#endif
	retainError(impl_->waitRingConsumers());
	impl_->freeRing();
	if (impl_->camera) {
		SDL_CloseCamera(impl_->camera);
		impl_->camera = nullptr;
	}
	impl_.reset();
	width_ = 0;
	height_ = 0;
	fps_ = 0;
	latestTimestampUs_ = 0;
	return firstError;
}
// ─── poll ────────────────────────────────────────────────────────────────────

bool oa::CameraCapture::poll() {
	if (impl_ and impl_->v4l2Streaming) {
		oa::VideoFrame frame;
		return pollFrame(frame);
	}
	if (!impl_ || !impl_->camera) return false;

	oa::U64 tsNs = 0;
	SDL_Surface* frame = SDL_AcquireCameraFrame(impl_->camera, &tsNs);
	if (!frame) return false;  // no new frame yet

	auto uploadAndRelease = [&](SDL_Surface* src) -> bool {
		auto& consumer = impl_->ringConsumers[static_cast<size_t>(impl_->head)];
		if (consumer.isValid() and not consumer.isComplete()) {
			SDL_ReleaseCameraFrame(impl_->camera, src);
			return false;
		}
		consumer = {};
		auto& dst = impl_->ring[static_cast<size_t>(impl_->head)];
		if (!dst.mappedPtr) {
			SDL_ReleaseCameraFrame(impl_->camera, src);
			return false;
		}
		const oa::I32 expectedPitch = impl_->w * 4;
		bool copied = false;
		if (src->format == SDL_PIXELFORMAT_RGBA32 && src->pitch == expectedPitch) {
			oa::memcpy(dst.mappedPtr, src->pixels,
				static_cast<oa::Usize>(impl_->h) * static_cast<oa::Usize>(expectedPitch));
			copied = true;
		} else {
			// convert to RGBA8 via SDL
			SDL_Surface* conv = SDL_ConvertSurface(src, SDL_PIXELFORMAT_RGBA32);
			if (conv) {
				for (oa::I32 y = 0; y < impl_->h; ++y) {
					const auto* source = static_cast<const oa::U8*>(conv->pixels)
						+ static_cast<size_t>(y) * static_cast<size_t>(conv->pitch);
					auto* destination = static_cast<oa::U8*>(dst.mappedPtr)
						+ static_cast<size_t>(y) * static_cast<size_t>(expectedPitch);
					oa::memcpy(destination, source, static_cast<oa::Usize>(expectedPitch));
				}
				copied = true;
				SDL_DestroySurface(conv);
			}
		}
		if (copied) {
			impl_->latest = impl_->head;
			impl_->head = (impl_->head + 1) % impl_->ringN;
		}
		SDL_ReleaseCameraFrame(impl_->camera, src);
		return copied;
	};

	if (not uploadAndRelease(frame)) return false;
	latestTimestampUs_ = tsNs / 1000ULL;
	return true;
}

bool oa::CameraCapture::pollFrame(oa::VideoFrame& outFrame) {
	if (impl_) {
#if defined(__linux__)
		if (impl_->v4l2Streaming) {
			if (not impl_->pollV4l2(outFrame, latestTimestampUs_)) {
				streaming_ = impl_->streaming;
				return false;
			}
			width_ = static_cast<oa::I32>(outFrame.width);
			height_ = static_cast<oa::I32>(outFrame.height);
			fps_ = impl_->fps;
			streaming_ = true;
			return true;
		}
		// A live V4L2 source may disappear temporarily. Retry with bounded
		// exponential backoff; the SDL fallback remains available at init.
		if (impl_->v4l2ReconnectEnabled
			and impl_->config.reconnectAttempts > 0U) {
			const auto now = oa::steadyNow();
			if (now >= impl_->nextReconnect
				and impl_->reconnects < impl_->config.reconnectAttempts) {
				const oa::U64 shift = oa::min<oa::U64>(impl_->reconnects, 5U);
				impl_->nextReconnect = now + oa::Duration::fromMilliseconds(
					static_cast<oa::I64>(impl_->config.reconnectBackoffMs * (1ULL << shift)));
				++impl_->reconnects;
				if (impl_->initV4l2()) {
					streaming_ = true;
					return false;
				}
			}
		}
#endif
	}
	if (!poll()) return false;
	const oavk::Buffer* buffer = (!impl_ || impl_->latest < 0)
		? nullptr
		: &impl_->ring[static_cast<size_t>(impl_->latest)];
	if (buffer == nullptr) return false;
	outFrame = {};
	outFrame.resource = oa::VideoFrameResource::Buffer;
	outFrame.buffer = buffer;
	outFrame.format = VK_FORMAT_R8G8B8A8_UNORM;
	outFrame.width = static_cast<oa::U32>(width_);
	outFrame.height = static_cast<oa::U32>(height_);
	outFrame.presentationTimestamp = latestTimestampUs_;
	outFrame.duration = fps_ > 0 ? 1'000'000ULL / static_cast<oa::U64>(fps_) : 0ULL;
	outFrame.isRgb = true;
	outFrame.colorSpace = oa::YCbCrModel::BT709;
	outFrame.fullRange = true;
	return true;
}

void oa::CameraCapture::release(const oa::VideoFrame& inFrame)
{
	release(inFrame, {});
}

void oa::CameraCapture::release(
	const oa::VideoFrame& inFrame,
	const oa::Event& inConsumed)
{
	if (not impl_) return;
	if (inFrame.resource == oa::VideoFrameResource::Buffer) {
		impl_->releaseRing(inFrame, inConsumed);
		return;
	}
#if defined(__linux__)
	if (inFrame.resource == oa::VideoFrameResource::Image) {
		impl_->releaseV4l2(inFrame, inConsumed);
	}
#else
	(void)inFrame;
	(void)inConsumed;
#endif
}

bool oa::CameraCapture::usesDmaBuf() const noexcept
{
#if defined(__linux__)
	return impl_ and impl_->v4l2Streaming;
#else
	return false;
#endif
}

oa::U64 oa::CameraCapture::formatGeneration() const noexcept
{
	return impl_ ? impl_->formatGen : 0U;
}

oa::U64 oa::CameraCapture::reconnectCount() const noexcept
{
	return impl_ ? impl_->reconnects : 0U;
}
