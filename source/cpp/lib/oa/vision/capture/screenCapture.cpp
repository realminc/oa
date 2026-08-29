#include <oa/vision/screenCapture.h>

#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/pair.h>
#include <oa/core/std/vector.h>
#include <oa/core/thread.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/externalMemory.h>
#include "../../core/logAccess.h"
#include "oa/runtime/engine/borrowedServiceRetirement.h"

#include <errno.h>
#include <string.h>

#if defined(__linux__)
#include <unistd.h>
#endif

#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
#include <libportal/portal.h>
#include <pipewire/pipewire.h>
#include <spa/buffer/buffer.h>
#include <spa/param/buffers.h>
#include <spa/param/format-utils.h>
#include <spa/param/video/format-utils.h>
#include <spa/param/video/raw-utils.h>
#endif

struct oa::ScreenCapture::Impl {
	oa::Engine* engine = nullptr;
	oa::ScreenCaptureConfig config = {};
	oa::Vector<oavk::Buffer> ring;
	oa::Vector<oa::Event> ringConsumers;
	oa::U32 head = 0;
	oa::U32 latest = 0;
	oa::U32 width = 0;
	oa::U32 height = 0;
	oa::U64 uploadedCpuSequence = 0;
	oa::Atomic<bool> streaming{false};

#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	XdpPortal* portal = nullptr;
	XdpSession* session = nullptr;
	GMainLoop* portalLoop = nullptr;
	GError* portalError = nullptr;
	int pipeWireFd = -1;
	oa::U32 pipeWireNode = PW_ID_ANY;

	pw_main_loop* pwLoop = nullptr;
	pw_context* pwContext = nullptr;
	pw_core* pwCore = nullptr;
	pw_stream* pwStream = nullptr;
	spa_hook streamListener = {};
	oa::Thread pwThread;

	oa::Mutex frameMutex;
	oa::Vector<oa::U8> cpuFrame;
	oa::U32 cpuWidth = 0;
	oa::U32 cpuHeight = 0;
	oa::U32 cpuStride = 0;
	oa::U32 sourceBytesPerPixel = 4;
	oa::U64 cpuTimestampUs = 0;
	oa::U64 cpuSequence = 0;
	spa_video_info_raw rawFormat = {};
	pw_buffer* heldDmaBuffer = nullptr;
	oa::ImportedDmaBufImage importedDmaImage;
	oa::U64 dmaSequence = 0;
	oa::U64 uploadedDmaSequence = 0;
	oa::U64 dmaTimestampUs = 0;
	int dmaFd = -1;
	oa::U64 dmaOffset = 0;
	oa::U64 dmaRowPitch = 0;
	struct PendingDmaRelease {
		pw_buffer* buffer = nullptr;
		oa::ImportedDmaBufImage imported;
		oa::Event consumed;
	};
	oa::Vector<PendingDmaRelease> pendingDmaReleases;
#endif

	void freeRing() {
		if (engine == nullptr) return;
		for (auto& buffer : ring) {
			if (buffer.buffer != VK_NULL_HANDLE) {
				oa::EngineBindlessAccess::deregisterBuffer(*engine, buffer);
				oa::EngineResourceAccess::freeBuffer(*engine, buffer);
			}
		}
		ring.clear();
		ringConsumers.clear();
		head = 0;
		latest = 0;
	}

	[[nodiscard]] bool ringConsumersComplete() const {
		for (const auto& consumer : ringConsumers) {
			if (consumer.isValid() and not consumer.isComplete()) return false;
		}
		return true;
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

	oa::Status ensureRing(oa::U32 inWidth, oa::U32 inHeight) {
		if (width == inWidth and height == inHeight and ring.size() > 0U) {
			return oa::Status::ok();
		}
		if (not ringConsumersComplete()) {
			return oa::Status::error(oa::StatusCode::Unavailable,
				"screen capture ring reconfiguration is waiting for GPU consumers");
		}
		freeRing();
		width = inWidth;
		height = inHeight;
		const oa::U32 ringCount = oa::max(2U, config.ringFrames);
		const oa::U64 bytes = static_cast<oa::U64>(width) * height * 4ULL;
		ring.resize(ringCount);
		ringConsumers.resize(ringCount);
		for (oa::U32 index = 0; index < ringCount; ++index) {
			auto result = oa::EngineResourceAccess::allocBuffer(*engine, bytes);
			if (not result.isOk()) {
				freeRing();
				return result.getStatus();
			}
			ring[index] = oa::move(*result);
			if (oa::EngineBindlessAccess::registerBuffer(
				*engine,
				ring[index]) == OA_BINDLESS_INVALID)
			{
				freeRing();
				return oa::Status::error(
					oa::StatusCode::ResourceExhausted,
					"screen capture bindless ring registration failed");
			}
		}
		return oa::Status::ok();
	}
};

#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
namespace {

VkFormat captureVkFormat(spa_video_format inFormat) {
	switch (inFormat) {
		case SPA_VIDEO_FORMAT_RGBA:
		case SPA_VIDEO_FORMAT_RGBx: return VK_FORMAT_R8G8B8A8_UNORM;
		case SPA_VIDEO_FORMAT_BGRA:
		case SPA_VIDEO_FORMAT_BGRx: return VK_FORMAT_B8G8R8A8_UNORM;
		default: return VK_FORMAT_UNDEFINED;
	}
}

const char* captureFormatName(spa_video_format inFormat) {
	switch (inFormat) {
		case SPA_VIDEO_FORMAT_RGBA: return "RGBA";
		case SPA_VIDEO_FORMAT_RGBx: return "RGBx";
		case SPA_VIDEO_FORMAT_BGRA: return "BGRA";
		case SPA_VIDEO_FORMAT_BGRx: return "BGRx";
		default: return "unknown";
	}
}

bool hasEnabledExtension(const oavk::Device& inDevice, oa::StringView inName) {
	for (const auto& extension : inDevice.info.software.enabledDeviceExtensions) {
		if (extension == inName) return true;
	}
	return false;
}

bool isImportableCaptureModifier(
	const oa::Engine& inEngine, VkFormat inFormat, oa::U64 inModifier)
{
	VkPhysicalDeviceExternalImageFormatInfo externalInfo = {};
	externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
	externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {};
	modifierInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;
	modifierInfo.pNext = &externalInfo;
	modifierInfo.drmFormatModifier = inModifier;
	modifierInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkPhysicalDeviceImageFormatInfo2 formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
	formatInfo.pNext = &modifierInfo;
	formatInfo.format = inFormat;
	formatInfo.type = VK_IMAGE_TYPE_2D;
	formatInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	formatInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkExternalImageFormatProperties externalProperties = {};
	externalProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
	VkImageFormatProperties2 properties = {};
	properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
	properties.pNext = &externalProperties;
	const auto& device = oa::EngineDeviceAccess::get(inEngine);
	const VkResult result = device.instanceDispatch.vkGetPhysicalDeviceImageFormatProperties2(
		static_cast<VkPhysicalDevice>(device.physicalDevice),
		&formatInfo, &properties);
	return result == VK_SUCCESS
		and (externalProperties.externalMemoryProperties.externalMemoryFeatures
			& VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0U;
}

oa::Vector<oa::U64> captureDmaBufModifiers(
	const oa::Engine& inEngine, VkFormat inFormat)
{
	const auto& device = oa::EngineDeviceAccess::get(inEngine);
	if (not hasEnabledExtension(device, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
		or not hasEnabledExtension(device, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)
		or not hasEnabledExtension(device, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
		return {};
	}
	VkDrmFormatModifierPropertiesListEXT list = {};
	list.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT;
	VkFormatProperties2 properties = {};
	properties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
	properties.pNext = &list;
	device.instanceDispatch.vkGetPhysicalDeviceFormatProperties2(
		static_cast<VkPhysicalDevice>(device.physicalDevice), inFormat, &properties);
	if (list.drmFormatModifierCount == 0U) return {};
	oa::Vector<VkDrmFormatModifierPropertiesEXT> candidates(list.drmFormatModifierCount);
	list.pDrmFormatModifierProperties = candidates.data();
	device.instanceDispatch.vkGetPhysicalDeviceFormatProperties2(
		static_cast<VkPhysicalDevice>(device.physicalDevice), inFormat, &properties);

	oa::Vector<oa::U64> result;
	result.reserve(candidates.size());
	for (const auto& candidate : candidates) {
		if (candidate.drmFormatModifierPlaneCount != 1U
			or (candidate.drmFormatModifierTilingFeatures
				& VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0U) {
			continue;
		}
		if (isImportableCaptureModifier(
			inEngine, inFormat, candidate.drmFormatModifier)) {
			result.pushBack(candidate.drmFormatModifier);
		}
	}
	return result;
}

const spa_pod* addCaptureFormat(
	spa_pod_builder& inBuilder,
	spa_video_format inFormat,
	const oa::U64* inModifier,
	const spa_rectangle& inPreferred,
	const spa_rectangle& inMinimum,
	const spa_rectangle& inMaximum,
	const spa_fraction& inFps,
	const spa_fraction& inMinFps,
	const spa_fraction& inMaxFps)
{
	spa_pod_frame frame = {};
	spa_pod_builder_push_object(
		&inBuilder, &frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(&inBuilder,
		SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(inFormat),
		0);
	if (inModifier != nullptr) {
		spa_pod_builder_prop(
			&inBuilder, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY);
		spa_pod_builder_long(&inBuilder, static_cast<oa::I64>(*inModifier));
	}
	spa_pod_builder_add(&inBuilder,
		SPA_FORMAT_VIDEO_size,
		SPA_POD_CHOICE_RANGE_Rectangle(&inPreferred, &inMinimum, &inMaximum),
		SPA_FORMAT_VIDEO_framerate,
		SPA_POD_CHOICE_RANGE_Fraction(&inFps, &inMinFps, &inMaxFps),
		0);
	return static_cast<const spa_pod*>(spa_pod_builder_pop(&inBuilder, &frame));
}

int queueReleasedBuffer(spa_loop*, bool, oa::U32, const void* inData,
	size_t inSize, void* inUserData)
{
	if (inSize != sizeof(pw_buffer*)) return -EINVAL;
	auto* stream = static_cast<pw_stream*>(inUserData);
	auto* buffer = *static_cast<pw_buffer* const*>(inData);
	return pw_stream_queue_buffer(stream, buffer);
}

void returnDmaBuffer(oa::ScreenCapture::Impl& inImpl, pw_buffer* inBuffer) {
	if (inBuffer == nullptr or inImpl.pwLoop == nullptr or inImpl.pwStream == nullptr) return;
	const int result = pw_loop_invoke(
		pw_main_loop_get_loop(inImpl.pwLoop), queueReleasedBuffer, 0,
		&inBuffer, sizeof(inBuffer), true, inImpl.pwStream);
	if (result < 0) {
		OaLogError(oa::LogComponent::Vision,
			"PipeWire screen capture could not return DMA-BUF: {}",
			::strerror(-result));
	}
}

void portalStartDone(GObject*, GAsyncResult* inResult, gpointer inData) {
	auto* impl = static_cast<oa::ScreenCapture::Impl*>(inData);
	if (not xdp_session_start_finish(impl->session, inResult, &impl->portalError)) {
		g_main_loop_quit(impl->portalLoop);
		return;
	}
	GVariant* streams = xdp_session_get_streams(impl->session);
	if (streams == nullptr) {
		impl->portalError = g_error_new_literal(
			G_IO_ERROR, G_IO_ERROR_NOT_FOUND, "portal returned no screencast streams");
		g_main_loop_quit(impl->portalLoop);
		return;
	}
	GVariantIter iter;
	g_variant_iter_init(&iter, streams);
	GVariant* properties = nullptr;
	guint32 node = PW_ID_ANY;
	if (not g_variant_iter_next(&iter, "(u@a{sv})", &node, &properties)) {
		impl->portalError = g_error_new_literal(
			G_IO_ERROR, G_IO_ERROR_NOT_FOUND, "portal returned no screencast streams");
	} else {
		impl->pipeWireNode = node;
		g_variant_unref(properties);
		impl->pipeWireFd = xdp_session_open_pipewire_remote(impl->session);
		if (impl->pipeWireFd < 0) {
			impl->portalError = g_error_new_literal(
				G_IO_ERROR, G_IO_ERROR_FAILED, "Failed to open portal PipeWire remote");
		}
	}
	g_main_loop_quit(impl->portalLoop);
}

void portalSessionDone(GObject* inObject, GAsyncResult* inResult, gpointer inData) {
	auto* impl = static_cast<oa::ScreenCapture::Impl*>(inData);
	impl->session = xdp_portal_create_screencast_session_finish(
		XDP_PORTAL(inObject), inResult, &impl->portalError);
	if (impl->session == nullptr) {
		g_main_loop_quit(impl->portalLoop);
		return;
	}
	xdp_session_start(impl->session, nullptr, nullptr, portalStartDone, impl);
}

void streamStateChanged(void* inData, pw_stream_state, pw_stream_state inState, const char* inError) {
	auto* impl = static_cast<oa::ScreenCapture::Impl*>(inData);
	if (inState == PW_STREAM_STATE_STREAMING) impl->streaming = true;
	if (inState == PW_STREAM_STATE_ERROR or inState == PW_STREAM_STATE_UNCONNECTED) {
		impl->streaming = false;
		if (inError != nullptr) {
			OaLogError(oa::LogComponent::Vision, "PipeWire screen stream: {}", inError);
		}
	}
}

void streamParamChanged(void* inData, oa::U32 inId, const spa_pod* inParam) {
	auto* impl = static_cast<oa::ScreenCapture::Impl*>(inData);
	if (inId != SPA_PARAM_Format or inParam == nullptr) return;
	if (spa_format_video_raw_parse(inParam, &impl->rawFormat) < 0) return;
	switch (impl->rawFormat.format) {
		case SPA_VIDEO_FORMAT_RGBA:
		case SPA_VIDEO_FORMAT_RGBx:
		case SPA_VIDEO_FORMAT_BGRA:
		case SPA_VIDEO_FORMAT_BGRx:
			impl->sourceBytesPerPixel = 4U;
			break;
		default:
			pw_stream_set_error(impl->pwStream, -EINVAL,
				"OA screen capture negotiated an unsupported pixel format");
			return;
	}
	{
		oa::ScopedLock<oa::Mutex> lock(impl->frameMutex);
		impl->cpuWidth = impl->rawFormat.size.width;
		impl->cpuHeight = impl->rawFormat.size.height;
		impl->cpuStride = impl->cpuWidth * 4U;
	}
	if ((impl->rawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U) {
		OaLogInfo(oa::LogComponent::Vision,
			"PipeWire screen capture: {}x{} {} via DMA-BUF modifier 0x{:x}",
			impl->rawFormat.size.width, impl->rawFormat.size.height,
			captureFormatName(impl->rawFormat.format),
			static_cast<unsigned long long>(impl->rawFormat.modifier));
	} else {
		OaLogInfo(oa::LogComponent::Vision,
			"PipeWire screen capture: {}x{} {} via mapped memory",
			impl->rawFormat.size.width, impl->rawFormat.size.height,
			captureFormatName(impl->rawFormat.format));
	}

	oa::U8 storage[512];
	spa_pod_builder builder = SPA_POD_BUILDER_INIT(storage, sizeof(storage));
	const spa_pod* params[1];
	const oa::I32 dataTypes = (impl->rawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U
		? (1 << SPA_DATA_DmaBuf)
		: ((1 << SPA_DATA_MemPtr) | (1 << SPA_DATA_MemFd));
	params[0] = static_cast<const spa_pod*>(spa_pod_builder_add_object(
		&builder,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(4, 2, 8),
		SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
		SPA_PARAM_BUFFERS_size, SPA_POD_Int(static_cast<oa::I32>(impl->rawFormat.size.width * impl->sourceBytesPerPixel * impl->rawFormat.size.height)),
		SPA_PARAM_BUFFERS_stride, SPA_POD_Int(static_cast<oa::I32>(impl->rawFormat.size.width * impl->sourceBytesPerPixel)),
		SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(dataTypes)));
	pw_stream_update_params(impl->pwStream, params, 1);
}

void streamProcess(void* inData) {
	auto* impl = static_cast<oa::ScreenCapture::Impl*>(inData);
	pw_buffer* pwBuffer = pw_stream_dequeue_buffer(impl->pwStream);
	if (pwBuffer == nullptr) return;
	spa_buffer* buffer = pwBuffer->buffer;
	if (buffer == nullptr or buffer->n_datas == 0U) {
		pw_stream_queue_buffer(impl->pwStream, pwBuffer);
		return;
	}
	spa_data& data = buffer->datas[0];
	if (data.chunk == nullptr or impl->cpuWidth == 0U or impl->cpuHeight == 0U) {
		pw_stream_queue_buffer(impl->pwStream, pwBuffer);
		return;
	}
	if (data.type == SPA_DATA_DmaBuf and data.fd >= 0
		and (impl->rawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U) {
		oa::ScopedLock<oa::Mutex> lock(impl->frameMutex);
		if (impl->heldDmaBuffer != nullptr) {
			// One explicitly-held producer frame at a time. Returning additional
			// buffers keeps PipeWire live while the consumer finishes the held one.
			pw_stream_queue_buffer(impl->pwStream, pwBuffer);
			return;
		}
		impl->heldDmaBuffer = pwBuffer;
		impl->dmaFd = static_cast<int>(data.fd);
		impl->dmaOffset = static_cast<oa::U64>(data.mapoffset) + data.chunk->offset;
		impl->dmaRowPitch = data.chunk->stride > 0
			? static_cast<oa::U64>(data.chunk->stride)
			: static_cast<oa::U64>(impl->cpuWidth) * impl->sourceBytesPerPixel;
		impl->dmaTimestampUs = pw_stream_get_nsec(impl->pwStream) / 1000ULL;
		++impl->dmaSequence;
		return; // release() queues this producer-owned buffer.
	}
	if (data.data == nullptr) {
		pw_stream_queue_buffer(impl->pwStream, pwBuffer);
		return;
	}
	const oa::U8* source = static_cast<const oa::U8*>(data.data) + data.chunk->offset;
	const oa::U32 sourceStride = data.chunk->stride > 0
		? static_cast<oa::U32>(data.chunk->stride)
		: impl->cpuWidth * impl->sourceBytesPerPixel;
	const oa::U64 required = static_cast<oa::U64>(impl->cpuWidth) * impl->cpuHeight * 4ULL;
	{
		oa::ScopedLock<oa::Mutex> lock(impl->frameMutex);
		impl->cpuFrame.resize(static_cast<oa::Usize>(required));
		const bool swapRedBlue = impl->rawFormat.format == SPA_VIDEO_FORMAT_BGRA
			or impl->rawFormat.format == SPA_VIDEO_FORMAT_BGRx;
		const bool forceAlpha = impl->rawFormat.format == SPA_VIDEO_FORMAT_RGBx
			or impl->rawFormat.format == SPA_VIDEO_FORMAT_BGRx;
		for (oa::U32 y = 0; y < impl->cpuHeight; ++y) {
			oa::U8* destination = impl->cpuFrame.data() + static_cast<oa::U64>(y) * impl->cpuWidth * 4ULL;
			const oa::U8* row = source + static_cast<oa::U64>(y) * sourceStride;
			if (not swapRedBlue and not forceAlpha) {
				oa::memcpy(destination, row, static_cast<oa::Usize>(impl->cpuWidth) * 4U);
				continue;
			}
			for (oa::U32 x = 0; x < impl->cpuWidth; ++x) {
				const oa::U8* pixel = row + x * 4U;
				oa::U8* out = destination + x * 4U;
				out[0] = swapRedBlue ? pixel[2] : pixel[0];
				out[1] = pixel[1];
				out[2] = swapRedBlue ? pixel[0] : pixel[2];
				out[3] = forceAlpha ? 255U : pixel[3];
			}
		}
		impl->cpuTimestampUs = pw_stream_get_nsec(impl->pwStream) / 1000ULL;
		++impl->cpuSequence;
	}
	pw_stream_queue_buffer(impl->pwStream, pwBuffer);
}

const pw_stream_events kStreamEvents = [] {
	pw_stream_events events = {};
	events.version = PW_VERSION_STREAM_EVENTS;
	events.state_changed = streamStateChanged;
	events.param_changed = streamParamChanged;
	events.process = streamProcess;
	return events;
}();

} // namespace
#endif

oa::ScreenCapture::ScreenCapture() = default;

oa::ScreenCapture::ScreenCapture(oa::ScreenCapture&& inOther) noexcept
	: impl_(oa::move(inOther.impl_)) {}

oa::ScreenCapture& oa::ScreenCapture::operator=(oa::ScreenCapture&& inOther) noexcept {
	if (this != &inOther) {
		abandon_();
		impl_ = oa::move(inOther.impl_);
	}
	return *this;
}
oa::ScreenCapture::~ScreenCapture() { abandon_(); }

void oa::ScreenCapture::abandon_() noexcept {
	if (not impl_) return;
	oa::Engine* engine = impl_->engine;
	if (engine == nullptr) {
		impl_.reset();
		return;
	}
#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	impl_->streaming = false;
	if (impl_->pwThread.joinable() and impl_->pwLoop != nullptr) {
		(void)pw_main_loop_quit(impl_->pwLoop);
	}
#endif
	auto retired = oa::makeUnique<oa::ScreenCapture>(oa::move(*this));
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retired.release(),
		&oa::ScreenCapture::completeRetired_,
		&oa::ScreenCapture::releaseRetired_);
}

oa::Status oa::ScreenCapture::completeRetired_(void* inPayload) {
	auto* capture = static_cast<oa::ScreenCapture*>(inPayload);
	return capture ? capture->close() : oa::Status::ok();
}

void oa::ScreenCapture::releaseRetired_(void* inPayload) {
	oa::UniquePtr<oa::ScreenCapture> capture(
		static_cast<oa::ScreenCapture*>(inPayload));
}

bool oa::ScreenCapture::isSupported() noexcept {
#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	return true;
#else
	return false;
#endif
}

static oa::Status closeScreenCaptureAfterOpenFailure(
	oa::ScreenCapture& inCapture,
	const oa::Status& inOpenFailure)
{
	const oa::Status closeStatus = inCapture.close();
	if (closeStatus.isOk()) return inOpenFailure;

	oa::String message = "screen capture open failed: ";
	message += inOpenFailure.toString();
	message += "; cleanup also failed: ";
	message += closeStatus.toString();
	return oa::Status::error(closeStatus.getCode(), oa::move(message));
}

oa::Result<oa::ScreenCapture> oa::ScreenCapture::open(
	oa::Engine& inEngine,
	const oa::ScreenCaptureConfig& inConfig)
{
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)inEngine;
	(void)inConfig;
	return oa::Status::error(oa::StatusCode::Unavailable,
		"Wayland screen capture requires libportal and PipeWire");
#else
	oa::ScreenCapture capture;
	capture.impl_ = oa::makeUnique<Impl>();
	auto& impl = *capture.impl_;
	impl.engine = &inEngine;
	impl.config = inConfig;
	impl.portal = xdp_portal_new();
	impl.portalLoop = g_main_loop_new(nullptr, FALSE);
	if (impl.portal == nullptr or impl.portalLoop == nullptr) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::Unavailable,
			"Could not initialize the desktop screencast portal"));
	}

	XdpOutputType output = static_cast<XdpOutputType>(XDP_OUTPUT_MONITOR | XDP_OUTPUT_WINDOW);
	if (inConfig.target == oa::ScreenCaptureTarget::Monitor) output = XDP_OUTPUT_MONITOR;
	if (inConfig.target == oa::ScreenCaptureTarget::Window) output = XDP_OUTPUT_WINDOW;
	const XdpCursorMode cursor = inConfig.cursor == oa::ScreenCaptureCursor::Embedded
		? XDP_CURSOR_MODE_EMBEDDED : XDP_CURSOR_MODE_HIDDEN;
	xdp_portal_create_screencast_session(
		impl.portal, output, XDP_SCREENCAST_FLAG_NONE, cursor,
		XDP_PERSIST_MODE_TRANSIENT, nullptr, nullptr, portalSessionDone, &impl);
	g_main_loop_run(impl.portalLoop);
	if (impl.portalError != nullptr) {
		oa::String message = impl.portalError->message;
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::PermissionDenied, oa::move(message)));
	}

	pw_init(nullptr, nullptr);
	impl.pwLoop = pw_main_loop_new(nullptr);
	if (impl.pwLoop == nullptr) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::Unavailable,
			"Could not create the PipeWire event loop"));
	}
	impl.pwContext = pw_context_new(pw_main_loop_get_loop(impl.pwLoop), nullptr, 0);
	if (impl.pwContext == nullptr) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::Unavailable,
			"Could not create the PipeWire context"));
	}
	impl.pwCore = pw_context_connect_fd(impl.pwContext, impl.pipeWireFd, nullptr, 0);
	if (impl.pwCore == nullptr) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::ConnectionFailed,
			"Could not connect to the portal PipeWire remote"));
	}
	impl.pipeWireFd = -1; // ownership transferred to pw_core
	impl.pwStream = pw_stream_new(
		impl.pwCore, "oa-screen-capture",
		pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Video",
			PW_KEY_MEDIA_CATEGORY, "capture",
			PW_KEY_MEDIA_ROLE, "Screen",
			nullptr));
	if (impl.pwStream == nullptr) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::Unavailable,
			"Could not create the PipeWire screen stream"));
	}
	pw_stream_add_listener(impl.pwStream, &impl.streamListener, &kStreamEvents, &impl);

	const spa_rectangle preferred = SPA_RECTANGLE(inConfig.preferredWidth, inConfig.preferredHeight);
	const spa_rectangle minimum = SPA_RECTANGLE(1, 1);
	const spa_rectangle maximum = SPA_RECTANGLE(16384, 16384);
	const spa_fraction fps = SPA_FRACTION(inConfig.preferredFps, 1);
	const spa_fraction minFps = SPA_FRACTION(0, 1);
	const spa_fraction maxFps = SPA_FRACTION(240, 1);
	struct CaptureFormat {
		spa_video_format spa;
		VkFormat vk;
	};
	constexpr CaptureFormat formats[] = {
		{ SPA_VIDEO_FORMAT_RGBA, VK_FORMAT_R8G8B8A8_UNORM },
		{ SPA_VIDEO_FORMAT_RGBx, VK_FORMAT_R8G8B8A8_UNORM },
		{ SPA_VIDEO_FORMAT_BGRA, VK_FORMAT_B8G8R8A8_UNORM },
		{ SPA_VIDEO_FORMAT_BGRx, VK_FORMAT_B8G8R8A8_UNORM },
	};
	oa::Vector<oa::Pair<spa_video_format, oa::U64>> dmaFormats;
	for (const auto& format : formats) {
		for (const oa::U64 modifier : captureDmaBufModifiers(inEngine, format.vk)) {
			dmaFormats.emplaceBack(format.spa, modifier);
		}
	}
	OaLogInfo(oa::LogComponent::Vision,
		"PipeWire screen capture: advertising {} DMA-BUF format/modifier pairs and mapped fallback",
		dmaFormats.size());
	// Each pod is small, but use dynamically-sized stable storage so every
	// importable modifier can be offered without a fixed stack limit.
	oa::Vector<oa::U8> storage(2048U + (dmaFormats.size() + 4U) * 512U);
	spa_pod_builder builder = SPA_POD_BUILDER_INIT(
		storage.data(), static_cast<oa::U32>(storage.size()));
	oa::Vector<const spa_pod*> params;
	params.reserve(dmaFormats.size() + 4U);
	for (const auto& [format, modifier] : dmaFormats) {
		params.pushBack(addCaptureFormat(
			builder, format, &modifier, preferred, minimum, maximum,
			fps, minFps, maxFps));
	}
	// mapped-memory formats are the universal fallback when the compositor
	// cannot allocate a modifier shared with the vulkan device.
	for (const auto& format : formats) {
		params.pushBack(addCaptureFormat(
			builder, format.spa, nullptr, preferred, minimum, maximum,
			fps, minFps, maxFps));
	}
	const int result = pw_stream_connect(
		impl.pwStream, PW_DIRECTION_INPUT, impl.pipeWireNode,
		static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS),
		params.data(), static_cast<oa::U32>(params.size()));
	if (result < 0) {
		return closeScreenCaptureAfterOpenFailure(capture, oa::Status::error(
			oa::StatusCode::ConnectionFailed,
			"Could not connect the PipeWire screen stream"));
	}
	const oa::LogSelection logSelection = oa::LogAccess::currentSelection();
	auto pipeWireThread = oa::Thread::create([loop = impl.pwLoop, logSelection] {
		oa::LogAccess::Scope logScope(logSelection);
		pw_main_loop_run(loop);
	});
	if (pipeWireThread.isError()) {
		return closeScreenCaptureAfterOpenFailure(
			capture, pipeWireThread.getStatus());
	}
	impl.pwThread = oa::move(*pipeWireThread);
	return oa::Result<oa::ScreenCapture>(oa::move(capture));
#endif
}

bool oa::ScreenCapture::poll(oa::VideoFrame& outFrame) {
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)outFrame;
	return false;
#else
	if (not impl_) return false;
	auto& impl = *impl_;
	// Retire producer buffers whose GPU consumers have completed. This keeps
	// the capture and encode queues fully asynchronous without letting
	// PipeWire overwrite a still-sampled DMA-BUF.
	for (oa::Usize index = 0U; index < impl.pendingDmaReleases.size();) {
		auto& pending = impl.pendingDmaReleases[index];
		if (not pending.consumed.isComplete()) { ++index; continue; }
		pending.imported = {};
		returnDmaBuffer(impl, pending.buffer);
		impl.pendingDmaReleases.erase(impl.pendingDmaReleases.data() + index);
	}
	pw_buffer* rejectedDmaBuffer = nullptr;
	{
		oa::ScopedLock<oa::Mutex> lock(impl.frameMutex);
		if (impl.heldDmaBuffer != nullptr
			and impl.dmaSequence != impl.uploadedDmaSequence) {
			oa::DmaBufImageDesc description;
			description.fd = impl.dmaFd;
			description.width = impl.cpuWidth;
			description.height = impl.cpuHeight;
			description.format = captureVkFormat(impl.rawFormat.format);
			description.modifier = impl.rawFormat.modifier;
			description.offset = impl.dmaOffset;
			description.rowPitch = impl.dmaRowPitch;
			auto imported = oa::ImportedDmaBufImage::import(*impl.engine, description);
			if (imported.isOk()) {
				impl.importedDmaImage = oa::move(*imported);
				impl.uploadedDmaSequence = impl.dmaSequence;
				impl.width = impl.cpuWidth;
				impl.height = impl.cpuHeight;

				outFrame = {};
				outFrame.resource = oa::VideoFrameResource::Image;
				outFrame.image = impl.importedDmaImage.image();
				outFrame.imageView = impl.importedDmaImage.view();
				outFrame.layout = VK_IMAGE_LAYOUT_GENERAL;
				outFrame.externalQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
				outFrame.format = impl.importedDmaImage.format();
				outFrame.width = impl.importedDmaImage.width();
				outFrame.height = impl.importedDmaImage.height();
				outFrame.presentationTimestamp = impl.dmaTimestampUs;
				outFrame.duration = impl.config.preferredFps > 0U
					? 1'000'000ULL / impl.config.preferredFps : 0ULL;
				outFrame.isRgb = true;
				outFrame.colorSpace = oa::YCbCrModel::BT709;
				outFrame.fullRange = true;
				return true;
			}
			OaLogError(oa::LogComponent::Vision,
				"PipeWire supplied an advertised DMA-BUF that vulkan could not import: {}",
				imported.getStatus().toString().cStr());
			rejectedDmaBuffer = impl.heldDmaBuffer;
			impl.heldDmaBuffer = nullptr;
			impl.dmaFd = -1;
			impl.dmaOffset = 0;
			impl.dmaRowPitch = 0;
			impl.uploadedDmaSequence = impl.dmaSequence;
		}
	}
	if (rejectedDmaBuffer != nullptr) {
		returnDmaBuffer(impl, rejectedDmaBuffer);
		return false;
	}

	oa::ScopedLock<oa::Mutex> lock(impl.frameMutex);
	if (impl.cpuSequence == 0U or impl.cpuSequence == impl.uploadedCpuSequence) return false;
	if (not impl.ensureRing(impl.cpuWidth, impl.cpuHeight).isOk()) return false;
	auto& consumer = impl.ringConsumers[impl.head];
	if (consumer.isValid() and not consumer.isComplete()) return false;
	consumer = {};
	auto& destination = impl.ring[impl.head];
	if (destination.mappedPtr == nullptr) return false;
	oa::memcpy(destination.mappedPtr, impl.cpuFrame.data(), impl.cpuFrame.size());
	impl.latest = impl.head;
	impl.head = (impl.head + 1U) % static_cast<oa::U32>(impl.ring.size());
	impl.uploadedCpuSequence = impl.cpuSequence;

	outFrame = {};
	outFrame.resource = oa::VideoFrameResource::Buffer;
	outFrame.buffer = &impl.ring[impl.latest];
	outFrame.format = VK_FORMAT_R8G8B8A8_UNORM;
	outFrame.width = impl.width;
	outFrame.height = impl.height;
	outFrame.presentationTimestamp = impl.cpuTimestampUs;
	outFrame.duration = impl.config.preferredFps > 0U
		? 1'000'000ULL / impl.config.preferredFps : 0ULL;
	outFrame.isRgb = true;
	outFrame.colorSpace = oa::YCbCrModel::BT709;
	outFrame.fullRange = true;
	return true;
#endif
}

void oa::ScreenCapture::release(const oa::VideoFrame& inFrame) {
	release(inFrame, {});
}

void oa::ScreenCapture::release(
	const oa::VideoFrame& inFrame,
	const oa::Event& inConsumed)
{
	if (not impl_) return;
	auto& impl = *impl_;
	if (inFrame.resource == oa::VideoFrameResource::Buffer) {
		impl.releaseRing(inFrame, inConsumed);
		return;
	}
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)inFrame;
	(void)inConsumed;
#else
	if (inFrame.resource != oa::VideoFrameResource::Image) return;
	pw_buffer* buffer = nullptr;
	oa::ImportedDmaBufImage imported;
	{
		oa::ScopedLock<oa::Mutex> lock(impl.frameMutex);
		if (not impl.importedDmaImage.isValid()
			or inFrame.image != impl.importedDmaImage.image()) return;
		imported = oa::move(impl.importedDmaImage);
		buffer = impl.heldDmaBuffer;
		impl.heldDmaBuffer = nullptr;
		impl.dmaFd = -1;
		impl.dmaOffset = 0;
		impl.dmaRowPitch = 0;
	}
	if (inConsumed.isValid() and not inConsumed.isComplete()) {
		Impl::PendingDmaRelease pending;
		pending.buffer = buffer;
		pending.imported = oa::move(imported);
		pending.consumed = inConsumed;
		impl.pendingDmaReleases.pushBack(oa::move(pending));
	} else {
		// destroy the import only after the final GPU consumer returned the
		// image to FOREIGN ownership, then recycle the producer buffer.
		imported = {};
		returnDmaBuffer(impl, buffer);
	}
#endif
}

oa::Status oa::ScreenCapture::close() {
	if (not impl_) return oa::Status::ok();
	auto& impl = *impl_;
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() and not inStatus.isOk()) firstError = inStatus;
	};
#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	impl.streaming = false;
	if (impl.pwThread.joinable() and impl.pwLoop != nullptr) {
		const int result = pw_main_loop_quit(impl.pwLoop);
		if (result < 0) {
			retainError(oa::Status::error(oa::StatusCode::Unavailable,
				"PipeWire screen capture loop could not be stopped"));
		}
	}
	if (impl.pwThread.joinable()) retainError(impl.pwThread.join());
	oa::ImportedDmaBufImage importedDmaImage;
	{
		oa::ScopedLock<oa::Mutex> lock(impl.frameMutex);
		importedDmaImage = oa::move(impl.importedDmaImage);
		impl.heldDmaBuffer = nullptr;
	}
	importedDmaImage = {};
	// PipeWire owns dequeued buffers until stream destruction. Once the producer
	// loop has stopped, no requeue is needed before destroying the stream.
	for (auto& pending : impl.pendingDmaReleases) {
		retainError(pending.consumed.wait());
		pending.imported = {};
	}
	impl.pendingDmaReleases.clear();
	if (impl.pwStream != nullptr) pw_stream_destroy(impl.pwStream);
	if (impl.pwCore != nullptr) pw_core_disconnect(impl.pwCore);
	if (impl.pwContext != nullptr) pw_context_destroy(impl.pwContext);
	if (impl.pwLoop != nullptr) pw_main_loop_destroy(impl.pwLoop);
	if (impl.session != nullptr) {
		xdp_session_close(impl.session);
		g_object_unref(impl.session);
	}
	if (impl.portal != nullptr) g_object_unref(impl.portal);
	if (impl.portalLoop != nullptr) g_main_loop_unref(impl.portalLoop);
	if (impl.portalError != nullptr) g_error_free(impl.portalError);
#if defined(__linux__)
	if (impl.pipeWireFd >= 0) ::close(impl.pipeWireFd);
#endif
#endif
	retainError(impl.waitRingConsumers());
	impl.freeRing();
	impl_.reset();
	return firstError;
}
bool oa::ScreenCapture::isStreaming() const noexcept {
	return impl_ and impl_->streaming.load();
}

oa::U32 oa::ScreenCapture::width() const noexcept { return impl_ ? impl_->width : 0U; }
oa::U32 oa::ScreenCapture::height() const noexcept { return impl_ ? impl_->height : 0U; }
