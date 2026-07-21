// OA Vision — OaCameraCapture implementation (SDL3 camera -> GPU ring buffer)
//
// SDL3 camera API:
//   SDL_GetCameras         — enumerate device IDs
//   SDL_OpenCamera         — open with requested spec
//   SDL_AcquireCameraFrame — returns SDL_Surface* (any format)
//   SDL_ReleaseCameraFrame — must be released before next acquire
//   SDL_ConvertSurface     — pixel-format conversion to RGBA8
//
// Ring buffer strategy:
//   N host-visible OaVkBuffers, each W*H*4 bytes.
//   Each buffer registered in the bindless heap so BindlessIndex
//   can be passed directly to OaUi::Image() — zero extra copy.
//   On Poll(): acquire surface → blit/convert to ring[head] → advance head.
//   LatestFrame() returns ring[latest], valid until next Poll().

#include <Oa/Vision/CameraCapture.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Std/UniquePtr.h>
#include <Oa/Runtime/ExternalMemory.h>
#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_camera.h>

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <chrono>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <libdrm/drm_fourcc.h>
#endif

// ─── OaCameraCapture::Impl ───────────────────────────────────────────────────

struct OaCameraCapture::Impl {
	OaEngine* Rt        = nullptr;
	SDL_Camera*        Camera    = nullptr;
	SDL_CameraSpec     Spec      = {};

	OaI32  W           = 0;
	OaI32  H           = 0;
	OaI32  Fps         = 0;
	OaI32  RingN       = 0;
	OaI32  Head        = 0;   // next buffer to write
	OaI32  Latest      = -1;  // last written buffer (-1 = no frame yet)
	bool   Streaming   = false;

	OaVec<OaVkBuffer> Ring;
	OaVec<OaCompletionToken> RingConsumers;
	OaCameraCaptureConfig Config = {};
	OaU64 FormatGen = 0U;
	OaU64 Reconnects = 0U;

#if defined(__linux__)
	struct V4l2Slot {
		OaImportedDmaBufImage Imported;
		int ExportFd = -1;
		OaU32 Index = 0U;
		bool Dequeued = false;
		bool ReleaseRequested = false;
		OaCompletionToken PendingConsumer;
	};
	int V4l2Fd = -1;
	OaVec<V4l2Slot> V4l2Slots;
	VkFormat V4l2VkFormat = VK_FORMAT_UNDEFINED;
	bool V4l2Streaming = false;
	bool V4l2ReconnectEnabled = false;
	OaU32 ConsecutiveFailures = 0U;
	std::chrono::steady_clock::time_point NextReconnect = {};

	bool InitV4l2();
	[[nodiscard]] OaStatus DestroyV4l2(bool InWaitConsumers = true);
	bool PollV4l2(OaVideoFrame& OutFrame, OaU64& OutTimestampUs);
	void RequeueCompletedV4l2();
	void ReleaseV4l2(const OaVideoFrame& InFrame, const OaCompletionToken& InConsumed);
#endif

	void FreeRing() {
		for (auto& b : Ring) {
			if (b.Buffer) {
				Rt->DeregisterBuffer(b);
				Rt->FreeBuffer(b);
			}
		}
		Ring.Clear();
		RingConsumers.Clear();
	}

	[[nodiscard]] OaStatus WaitRingConsumers() const {
		OaStatus firstError = OaStatus::Ok();
		for (const auto& consumer : RingConsumers) {
			const auto status = consumer.Wait();
			if (firstError.IsOk() and not status.IsOk()) firstError = status;
		}
		return firstError;
	}

	void ReleaseRing(
		const OaVideoFrame& InFrame,
		const OaCompletionToken& InConsumed)
	{
		if (InFrame.Buffer == nullptr) return;
		for (OaUsize index = 0U; index < Ring.Size(); ++index) {
			if (InFrame.Buffer != &Ring[index]) continue;
			RingConsumers[index] = InConsumed;
			return;
		}
	}
};

#if defined(__linux__)
namespace {

int V4l2Ioctl(int InFd, unsigned long InRequest, void* InArgument)
{
	int result;
	do { result = ::ioctl(InFd, InRequest, InArgument); }
	while (result < 0 and errno == EINTR);
	return result;
}

struct CameraPackedFormat {
	OaU32 V4l2 = 0U;
	VkFormat Vulkan = VK_FORMAT_UNDEFINED;
};

constexpr CameraPackedFormat kCameraFormats[] = {
	// On little-endian Linux these V4L2 layouts are byte-compatible with the
	// matching Vulkan UNORM formats. X/alpha is ignored by RGBA->NV12.
	{V4L2_PIX_FMT_XBGR32, VK_FORMAT_R8G8B8A8_UNORM},
	{V4L2_PIX_FMT_ABGR32, VK_FORMAT_R8G8B8A8_UNORM},
	{V4L2_PIX_FMT_XRGB32, VK_FORMAT_B8G8R8A8_UNORM},
	{V4L2_PIX_FMT_ARGB32, VK_FORMAT_B8G8R8A8_UNORM},
};

} // namespace

bool OaCameraCapture::Impl::InitV4l2()
{
	(void)DestroyV4l2(true);
	OaString path = Config.DevicePath;
	if (path.empty()) path = OaString("/dev/video" + std::to_string(Config.DeviceIndex));
	V4l2Fd = ::open(path.CStr(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
	if (V4l2Fd < 0) return false;

	v4l2_capability capabilities = {};
	if (V4l2Ioctl(V4l2Fd, VIDIOC_QUERYCAP, &capabilities) < 0
		or (capabilities.device_caps & V4L2_CAP_VIDEO_CAPTURE) == 0U
		or (capabilities.device_caps & V4L2_CAP_STREAMING) == 0U) {
		(void)DestroyV4l2(false);
		return false;
	}

	v4l2_format selected = {};
	selected.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bool found = false;
	for (const auto& candidate : kCameraFormats) {
		v4l2_format format = {};
		format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		format.fmt.pix.width = static_cast<OaU32>(Config.Width);
		format.fmt.pix.height = static_cast<OaU32>(Config.Height);
		format.fmt.pix.pixelformat = candidate.V4l2;
		format.fmt.pix.field = V4L2_FIELD_ANY;
		if (V4l2Ioctl(V4l2Fd, VIDIOC_S_FMT, &format) == 0
			and format.fmt.pix.pixelformat == candidate.V4l2) {
			selected = format;
			V4l2VkFormat = candidate.Vulkan;
			found = true;
			break;
		}
	}
	if (not found) { (void)DestroyV4l2(false); return false; }

	v4l2_streamparm parameters = {};
	parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	parameters.parm.capture.timeperframe.numerator = 1U;
	parameters.parm.capture.timeperframe.denominator = static_cast<OaU32>(Config.Fps);
	(void)V4l2Ioctl(V4l2Fd, VIDIOC_S_PARM, &parameters);
	v4l2_event_subscription subscription = {};
	subscription.type = V4L2_EVENT_SOURCE_CHANGE;
	(void)V4l2Ioctl(V4l2Fd, VIDIOC_SUBSCRIBE_EVENT, &subscription);

	v4l2_requestbuffers request = {};
	request.count = static_cast<OaU32>(std::max(2, Config.RingFrames));
	request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	request.memory = V4L2_MEMORY_MMAP;
	if (V4l2Ioctl(V4l2Fd, VIDIOC_REQBUFS, &request) < 0 or request.count < 2U) {
		(void)DestroyV4l2(false);
		return false;
	}

	V4l2Slots.Resize(request.count);
	for (OaU32 index = 0U; index < request.count; ++index) {
		v4l2_buffer buffer = {};
		buffer.type = request.type;
		buffer.memory = request.memory;
		buffer.index = index;
		if (V4l2Ioctl(V4l2Fd, VIDIOC_QUERYBUF, &buffer) < 0) {
			(void)DestroyV4l2(false); return false;
		}
		v4l2_exportbuffer exportBuffer = {};
		exportBuffer.type = request.type;
		exportBuffer.index = index;
		exportBuffer.flags = O_CLOEXEC;
		if (V4l2Ioctl(V4l2Fd, VIDIOC_EXPBUF, &exportBuffer) < 0) {
			(void)DestroyV4l2(false); return false;
		}
		auto& slot = V4l2Slots[index];
		slot.ExportFd = exportBuffer.fd;
		slot.Index = index;
		OaDmaBufImageDesc description;
		description.Fd = slot.ExportFd;
		description.Width = selected.fmt.pix.width;
		description.Height = selected.fmt.pix.height;
		description.Format = V4l2VkFormat;
		description.Modifier = DRM_FORMAT_MOD_LINEAR;
		description.RowPitch = selected.fmt.pix.bytesperline;
		auto imported = OaImportedDmaBufImage::Import(*Rt, description);
		if (not imported.IsOk()) {
			(void)DestroyV4l2(false); return false;
		}
		slot.Imported = OaStdMove(*imported);
		if (V4l2Ioctl(V4l2Fd, VIDIOC_QBUF, &buffer) < 0) {
			(void)DestroyV4l2(false); return false;
		}
	}
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (V4l2Ioctl(V4l2Fd, VIDIOC_STREAMON, &type) < 0) {
		(void)DestroyV4l2(false); return false;
	}
	V4l2Streaming = true;
	V4l2ReconnectEnabled = true;
	W = static_cast<OaI32>(selected.fmt.pix.width);
	H = static_cast<OaI32>(selected.fmt.pix.height);
	Fps = Config.Fps;
	if (parameters.parm.capture.timeperframe.numerator > 0U) {
		Fps = static_cast<OaI32>(parameters.parm.capture.timeperframe.denominator
			/ parameters.parm.capture.timeperframe.numerator);
	}
	if (Fps <= 0) Fps = Config.Fps;
	Streaming = true;
	ConsecutiveFailures = 0U;
	++FormatGen;
	OA_LOG_INFO(OaLogComponent::App,
		"OaCameraCapture: V4L2 DMA-BUF %s %dx%d @ %d fps, ring=%u",
		path.CStr(), W, H, Fps, request.count);
	return true;
}

OaStatus OaCameraCapture::Impl::DestroyV4l2(bool InWaitConsumers)
{
	OaStatus firstError = OaStatus::Ok();
	if (InWaitConsumers) {
		for (auto& slot : V4l2Slots) {
			if (not slot.ReleaseRequested) continue;
			const auto status = slot.PendingConsumer.Wait();
			if (firstError.IsOk() and not status.IsOk()) firstError = status;
		}
	}
	if (V4l2Fd >= 0 and V4l2Streaming) {
		v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		(void)V4l2Ioctl(V4l2Fd, VIDIOC_STREAMOFF, &type);
	}
	V4l2Streaming = false;
	for (auto& slot : V4l2Slots) {
		slot.Imported.Destroy();
		if (slot.ExportFd >= 0) ::close(slot.ExportFd);
		slot.ExportFd = -1;
	}
	V4l2Slots.Clear();
	if (V4l2Fd >= 0) ::close(V4l2Fd);
	V4l2Fd = -1;
	V4l2VkFormat = VK_FORMAT_UNDEFINED;
	return firstError;
}

void OaCameraCapture::Impl::RequeueCompletedV4l2()
{
	for (auto& slot : V4l2Slots) {
		if (not slot.Dequeued or not slot.ReleaseRequested
			or not slot.PendingConsumer.IsComplete()) continue;
		v4l2_buffer buffer = {};
		buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = slot.Index;
		if (V4l2Ioctl(V4l2Fd, VIDIOC_QBUF, &buffer) == 0) {
			slot.Dequeued = false;
			slot.ReleaseRequested = false;
			slot.PendingConsumer = {};
		}
	}
}

bool OaCameraCapture::Impl::PollV4l2(OaVideoFrame& OutFrame, OaU64& OutTimestampUs)
{
	RequeueCompletedV4l2();
	v4l2_event event = {};
	if (V4l2Ioctl(V4l2Fd, VIDIOC_DQEVENT, &event) == 0
		and event.type == V4L2_EVENT_SOURCE_CHANGE) {
		V4l2Streaming = false;
		Streaming = false;
		return false;
	}
	v4l2_buffer buffer = {};
	buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buffer.memory = V4L2_MEMORY_MMAP;
	if (V4l2Ioctl(V4l2Fd, VIDIOC_DQBUF, &buffer) < 0) {
		if (errno == EAGAIN) return false;
		++ConsecutiveFailures;
		V4l2Streaming = false;
		Streaming = false;
		return false;
	}
	if (buffer.index >= V4l2Slots.Size()) return false;
	auto& slot = V4l2Slots[buffer.index];
	slot.Dequeued = true;
	slot.ReleaseRequested = false;
	slot.PendingConsumer = {};
	OutTimestampUs = static_cast<OaU64>(buffer.timestamp.tv_sec) * 1'000'000ULL
		+ static_cast<OaU64>(buffer.timestamp.tv_usec);
	OutFrame = {};
	OutFrame.Resource = OaVideoFrameResource::Image;
	OutFrame.Image = slot.Imported.Image();
	OutFrame.ImageView = slot.Imported.View();
	OutFrame.Layout = VK_IMAGE_LAYOUT_GENERAL;
	OutFrame.ExternalQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
	OutFrame.Format = V4l2VkFormat;
	OutFrame.Width = slot.Imported.Width();
	OutFrame.Height = slot.Imported.Height();
	OutFrame.PresentationTimestamp = OutTimestampUs;
	OutFrame.Duration = Fps > 0 ? 1'000'000ULL / static_cast<OaU64>(Fps) : 0U;
	OutFrame.IsRgb = true;
	OutFrame.ColorSpace = OaYCbCrModel::BT709;
	OutFrame.FullRange = true;
	ConsecutiveFailures = 0U;
	return true;
}

void OaCameraCapture::Impl::ReleaseV4l2(
	const OaVideoFrame& InFrame, const OaCompletionToken& InConsumed)
{
	for (auto& slot : V4l2Slots) {
		if (slot.Dequeued and slot.Imported.Image() == InFrame.Image) {
			slot.ReleaseRequested = true;
			slot.PendingConsumer = InConsumed;
			RequeueCompletedV4l2();
			return;
		}
	}
}
#endif

// ─── move / dtor ─────────────────────────────────────────────────────────────

OaCameraCapture::OaCameraCapture(OaCameraCapture&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_))
	, Width_(InOther.Width_)
	, Height_(InOther.Height_)
	, Fps_(InOther.Fps_)
	, Streaming_(InOther.Streaming_)
	, LatestTimestampUs_(InOther.LatestTimestampUs_)
{
	InOther.Width_ = 0;
	InOther.Height_ = 0;
	InOther.Fps_ = 0;
	InOther.Streaming_ = false;
	InOther.LatestTimestampUs_ = 0;
}

OaCameraCapture& OaCameraCapture::operator=(OaCameraCapture&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_ = OaStdMove(InOther.Impl_);
		Width_ = InOther.Width_;
		Height_ = InOther.Height_;
		Fps_ = InOther.Fps_;
		Streaming_ = InOther.Streaming_;
		LatestTimestampUs_ = InOther.LatestTimestampUs_;
		InOther.Width_ = 0;
		InOther.Height_ = 0;
		InOther.Fps_ = 0;
		InOther.Streaming_ = false;
		InOther.LatestTimestampUs_ = 0;
	}
	return *this;
}

OaCameraCapture::~OaCameraCapture() {
	Abandon_();
}

void OaCameraCapture::Abandon_() noexcept {
	if (not Impl_) return;
	OaEngine* engine = Impl_->Rt;
	if (engine == nullptr) {
		Impl_.reset();
		return;
	}
	auto retired = OaMakeUniquePtr<OaCameraCapture>(OaStdMove(*this));
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retired.Release(),
		&OaCameraCapture::CompleteRetired_,
		&OaCameraCapture::ReleaseRetired_);
}

OaStatus OaCameraCapture::CompleteRetired_(void* InPayload) {
	auto* capture = static_cast<OaCameraCapture*>(InPayload);
	return capture ? capture->Close() : OaStatus::Ok();
}

void OaCameraCapture::ReleaseRetired_(void* InPayload) {
	OaUniquePtr<OaCameraCapture> capture(
		static_cast<OaCameraCapture*>(InPayload));
}

// ─── Init ────────────────────────────────────────────────────────────────────

OaStatus OaCameraCapture::Init(
	OaEngine& InRt, const OaCameraCaptureConfig& InConfig) {
	OA_RETURN_IF_ERROR(Close());
	if (InConfig.DeviceIndex < 0 or InConfig.Width <= 0 or InConfig.Height <= 0
		or InConfig.Fps <= 0 or InConfig.RingFrames < 2) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaCameraCapture requires a non-negative device, positive extent/fps, and ring >= 2");
	}
	Impl_ = OaStdMakeUnique<Impl>();
	Impl_->Rt   = &InRt;
	Impl_->Config = InConfig;
	Impl_->W    = InConfig.Width;
	Impl_->H    = InConfig.Height;
	Impl_->Fps  = InConfig.Fps;
	Impl_->RingN = std::max(2, InConfig.RingFrames);

#if defined(__linux__)
	if (InConfig.PreferDmaBuf and Impl_->InitV4l2()) {
		Width_ = Impl_->W;
		Height_ = Impl_->H;
		Fps_ = Impl_->Fps;
		Streaming_ = true;
		return OaStatus::Ok();
	}
	if (InConfig.PreferDmaBuf) {
		OA_LOG_INFO(OaLogComponent::App,
			"OaCameraCapture: V4L2 DMA-BUF unavailable; using SDL mapped fallback");
	}
#endif

	if (!SDL_InitSubSystem(SDL_INIT_CAMERA)) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_INIT_CAMERA failed: %s", SDL_GetError());
		return OaStatus::Error("OaCameraCapture: SDL_INIT_CAMERA failed");
	}

	int numCams = 0;
	SDL_CameraID* camIds = SDL_GetCameras(&numCams);
	if (!camIds || numCams == 0) {
		SDL_free(camIds);
		return OaStatus::Error("OaCameraCapture: no cameras found");
	}

	OaI32 devIdx = std::min(InConfig.DeviceIndex, numCams - 1);
	SDL_CameraID id = camIds[devIdx];
	SDL_free(camIds);

	SDL_CameraSpec desired = {};
	desired.format     = SDL_PIXELFORMAT_RGBA32;
	desired.width      = InConfig.Width;
	desired.height     = InConfig.Height;
	desired.framerate_numerator   = InConfig.Fps;
	desired.framerate_denominator = 1;

	Impl_->Camera = SDL_OpenCamera(id, &desired);
	if (!Impl_->Camera) {
		OA_LOG_ERROR(OaLogComponent::App, "SDL_OpenCamera failed: %s", SDL_GetError());
		return OaStatus::Error("OaCameraCapture: SDL_OpenCamera failed");
	}

	// Wait for camera permission grant (SDL3 requires explicit poll loop)
	// Camera permission is asynchronous. Keep the deadline finite so a broken
	// backend cannot hang a headless process forever.
	bool permissionResolved = false;
	for (OaI32 retry = 0; retry < 5000; ++retry) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_EVENT_CAMERA_DEVICE_APPROVED) {
				OA_LOG_INFO(OaLogComponent::App,
					"OaCameraCapture: camera approved (device %d)", devIdx);
				permissionResolved = true;
				break;
			}
			if (e.type == SDL_EVENT_CAMERA_DEVICE_DENIED) {
				SDL_CloseCamera(Impl_->Camera);
				Impl_->Camera = nullptr;
				return OaStatus::Error("OaCameraCapture: camera access denied");
			}
		}
		if (permissionResolved) break;
		SDL_Delay(1);
	}
	if (not permissionResolved) {
		Destroy();
		return OaStatus::Error(OaStatusCode::DeadlineExceeded,
			"OaCameraCapture: camera permission timed out");
	}

	// Read actual format after approval
	if (SDL_GetCameraFormat(Impl_->Camera, &Impl_->Spec)) {
		Impl_->W   = Impl_->Spec.width;
		Impl_->H   = Impl_->Spec.height;
		if (Impl_->Spec.framerate_denominator > 0) {
			const OaI64 numerator = Impl_->Spec.framerate_numerator;
			const OaI64 denominator = Impl_->Spec.framerate_denominator;
			Impl_->Fps = static_cast<OaI32>((numerator + denominator / 2) / denominator);
		}
		if (Impl_->Fps <= 0) Impl_->Fps = InConfig.Fps;
		OA_LOG_INFO(OaLogComponent::App,
			"OaCameraCapture: opened %dx%d @ %d fps (format=0x%X)",
			Impl_->W, Impl_->H, Impl_->Fps,
			static_cast<OaU32>(Impl_->Spec.format));
	}

	// Allocate GPU ring buffers (host-visible for direct memcpy)
	OaU64 frameBytes = static_cast<OaU64>(Impl_->W)
		* static_cast<OaU64>(Impl_->H) * 4ULL;
	Impl_->Ring.Resize(static_cast<OaUsize>(Impl_->RingN));
	Impl_->RingConsumers.Resize(static_cast<OaUsize>(Impl_->RingN));
	for (OaI32 i = 0; i < Impl_->RingN; ++i) {
		auto res = InRt.AllocBuffer(frameBytes);
		if (!res.IsOk()) {
			// Free already-allocated ring entries
			for (OaI32 j = 0; j < i; ++j) {
				auto& b = Impl_->Ring[static_cast<size_t>(j)];
				InRt.DeregisterBuffer(b);
				InRt.FreeBuffer(b);
			}
			OA_LOG_ERROR(OaLogComponent::App, "Ring buffer alloc failed: %s",
				res.GetStatus().ToString().c_str());
			return OaStatus::Error("OaCameraCapture: ring buffer alloc failed");
		}
		Impl_->Ring[static_cast<size_t>(i)] = std::move(res.GetValue());
		InRt.RegisterBuffer(Impl_->Ring[static_cast<size_t>(i)]);
	}

	Width_     = Impl_->W;
	Height_    = Impl_->H;
	Fps_       = Impl_->Fps;
	Streaming_ = true;

	OA_LOG_INFO(OaLogComponent::App,
		"OaCameraCapture: ring x%d, %.1f MB/frame",
		Impl_->RingN, frameBytes / 1e6f);

	return OaStatus::Ok();
}

// ─── Destroy ─────────────────────────────────────────────────────────────────

OaStatus OaCameraCapture::Close() {
	if (not Impl_) return OaStatus::Ok();
	OaStatus firstError = OaStatus::Ok();
	auto retainError = [&firstError](const OaStatus& InStatus) {
		if (firstError.IsOk() and not InStatus.IsOk()) firstError = InStatus;
	};
	Streaming_ = false;
#if defined(__linux__)
	retainError(Impl_->DestroyV4l2());
#endif
	retainError(Impl_->WaitRingConsumers());
	Impl_->FreeRing();
	if (Impl_->Camera) {
		SDL_CloseCamera(Impl_->Camera);
		Impl_->Camera = nullptr;
	}
	Impl_.reset();
	Width_ = 0;
	Height_ = 0;
	Fps_ = 0;
	LatestTimestampUs_ = 0;
	return firstError;
}

void OaCameraCapture::Destroy() {
	if (const auto status = Close(); not status.IsOk()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"OaCameraCapture::Destroy: shutdown failed: %s",
			status.ToString().c_str());
	}
}

// ─── Poll ────────────────────────────────────────────────────────────────────

bool OaCameraCapture::Poll() {
	if (Impl_ and Impl_->V4l2Streaming) {
		OaVideoFrame frame;
		return PollFrame(frame);
	}
	if (!Impl_ || !Impl_->Camera) return false;

	OaU64 tsNs = 0;
	SDL_Surface* frame = SDL_AcquireCameraFrame(Impl_->Camera, &tsNs);
	if (!frame) return false;  // no new frame yet

	auto UploadAndRelease = [&](SDL_Surface* Src) -> bool {
		auto& consumer = Impl_->RingConsumers[static_cast<size_t>(Impl_->Head)];
		if (consumer.IsValid() and not consumer.IsComplete()) {
			SDL_ReleaseCameraFrame(Impl_->Camera, Src);
			return false;
		}
		consumer = {};
		auto& dst = Impl_->Ring[static_cast<size_t>(Impl_->Head)];
		if (!dst.MappedPtr) {
			SDL_ReleaseCameraFrame(Impl_->Camera, Src);
			return false;
		}
		const OaI32 expectedPitch = Impl_->W * 4;
		bool copied = false;
		if (Src->format == SDL_PIXELFORMAT_RGBA32 && Src->pitch == expectedPitch) {
			std::memcpy(dst.MappedPtr, Src->pixels,
				static_cast<size_t>(Impl_->H) * static_cast<size_t>(expectedPitch));
			copied = true;
		} else {
			// Convert to RGBA8 via SDL
			SDL_Surface* conv = SDL_ConvertSurface(Src, SDL_PIXELFORMAT_RGBA32);
			if (conv) {
				for (OaI32 y = 0; y < Impl_->H; ++y) {
					const auto* source = static_cast<const OaU8*>(conv->pixels)
						+ static_cast<size_t>(y) * static_cast<size_t>(conv->pitch);
					auto* destination = static_cast<OaU8*>(dst.MappedPtr)
						+ static_cast<size_t>(y) * static_cast<size_t>(expectedPitch);
					std::memcpy(destination, source, static_cast<size_t>(expectedPitch));
				}
				copied = true;
				SDL_DestroySurface(conv);
			}
		}
		if (copied) {
			Impl_->Latest = Impl_->Head;
			Impl_->Head = (Impl_->Head + 1) % Impl_->RingN;
		}
		SDL_ReleaseCameraFrame(Impl_->Camera, Src);
		return copied;
	};

	if (not UploadAndRelease(frame)) return false;
	LatestTimestampUs_ = tsNs / 1000ULL;
	return true;
}

bool OaCameraCapture::PollFrame(OaVideoFrame& OutFrame) {
	if (Impl_) {
#if defined(__linux__)
		if (Impl_->V4l2Streaming) {
			if (not Impl_->PollV4l2(OutFrame, LatestTimestampUs_)) {
				Streaming_ = Impl_->Streaming;
				return false;
			}
			Width_ = static_cast<OaI32>(OutFrame.Width);
			Height_ = static_cast<OaI32>(OutFrame.Height);
			Fps_ = Impl_->Fps;
			Streaming_ = true;
			return true;
		}
		// A live V4L2 source may disappear temporarily. Retry with bounded
		// exponential backoff; the SDL fallback remains available at Init.
		if (Impl_->V4l2ReconnectEnabled
			and Impl_->Config.ReconnectAttempts > 0U) {
			const auto now = std::chrono::steady_clock::now();
			if (now >= Impl_->NextReconnect
				and Impl_->Reconnects < Impl_->Config.ReconnectAttempts) {
				const OaU64 shift = std::min<OaU64>(Impl_->Reconnects, 5U);
				Impl_->NextReconnect = now + std::chrono::milliseconds(
					Impl_->Config.ReconnectBackoffMs * (1ULL << shift));
				++Impl_->Reconnects;
				if (Impl_->InitV4l2()) {
					Streaming_ = true;
					return false;
				}
			}
		}
#endif
	}
	if (!Poll()) return false;
	const OaVkBuffer* buffer = LatestFrame();
	if (buffer == nullptr) return false;
	OutFrame = {};
	OutFrame.Resource = OaVideoFrameResource::Buffer;
	OutFrame.Buffer = buffer;
	OutFrame.Format = VK_FORMAT_R8G8B8A8_UNORM;
	OutFrame.Width = static_cast<OaU32>(Width_);
	OutFrame.Height = static_cast<OaU32>(Height_);
	OutFrame.PresentationTimestamp = LatestTimestampUs_;
	OutFrame.Duration = Fps_ > 0 ? 1'000'000ULL / static_cast<OaU64>(Fps_) : 0ULL;
	OutFrame.IsRgb = true;
	OutFrame.ColorSpace = OaYCbCrModel::BT709;
	OutFrame.FullRange = true;
	return true;
}

void OaCameraCapture::Release(const OaVideoFrame& InFrame)
{
	Release(InFrame, {});
}

void OaCameraCapture::Release(
	const OaVideoFrame& InFrame,
	const OaCompletionToken& InConsumed)
{
	if (not Impl_) return;
	if (InFrame.Resource == OaVideoFrameResource::Buffer) {
		Impl_->ReleaseRing(InFrame, InConsumed);
		return;
	}
#if defined(__linux__)
	if (InFrame.Resource == OaVideoFrameResource::Image) {
		Impl_->ReleaseV4l2(InFrame, InConsumed);
	}
#else
	(void)InFrame;
	(void)InConsumed;
#endif
}

bool OaCameraCapture::UsesDmaBuf() const noexcept
{
#if defined(__linux__)
	return Impl_ and Impl_->V4l2Streaming;
#else
	return false;
#endif
}

OaU64 OaCameraCapture::FormatGeneration() const noexcept
{
	return Impl_ ? Impl_->FormatGen : 0U;
}

OaU64 OaCameraCapture::ReconnectCount() const noexcept
{
	return Impl_ ? Impl_->Reconnects : 0U;
}

// ─── LatestFrame ─────────────────────────────────────────────────────────────

OaVkBuffer* OaCameraCapture::LatestFrame() noexcept {
	if (!Impl_ || Impl_->Latest < 0) return nullptr;
	return &Impl_->Ring[static_cast<size_t>(Impl_->Latest)];
}

const OaVkBuffer* OaCameraCapture::LatestFrame() const noexcept {
	if (!Impl_ || Impl_->Latest < 0) return nullptr;
	return &Impl_->Ring[static_cast<size_t>(Impl_->Latest)];
}
