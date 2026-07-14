#include <Oa/Vision/ScreenCapture.h>

#include <Oa/Core/Log.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/ExternalMemory.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

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

struct OaScreenCapture::Impl {
	OaComputeEngine* Engine = nullptr;
	OaScreenCaptureConfig Config = {};
	OaVec<OaVkBuffer> Ring;
	OaU32 Head = 0;
	OaU32 Latest = 0;
	OaU32 Width = 0;
	OaU32 Height = 0;
	OaU64 UploadedCpuSequence = 0;
	std::atomic<bool> Streaming = false;

#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	XdpPortal* Portal = nullptr;
	XdpSession* Session = nullptr;
	GMainLoop* PortalLoop = nullptr;
	GError* PortalError = nullptr;
	int PipeWireFd = -1;
	OaU32 PipeWireNode = PW_ID_ANY;

	pw_main_loop* PwLoop = nullptr;
	pw_context* PwContext = nullptr;
	pw_core* PwCore = nullptr;
	pw_stream* PwStream = nullptr;
	spa_hook StreamListener = {};
	std::thread PwThread;

	std::mutex FrameMutex;
	OaVec<OaU8> CpuFrame;
	OaU32 CpuWidth = 0;
	OaU32 CpuHeight = 0;
	OaU32 CpuStride = 0;
	OaU32 SourceBytesPerPixel = 4;
	OaU64 CpuTimestampUs = 0;
	OaU64 CpuSequence = 0;
	spa_video_info_raw RawFormat = {};
	pw_buffer* HeldDmaBuffer = nullptr;
	OaImportedDmaBufImage ImportedDmaImage;
	OaU64 DmaSequence = 0;
	OaU64 UploadedDmaSequence = 0;
	OaU64 DmaTimestampUs = 0;
	int DmaFd = -1;
	OaU64 DmaOffset = 0;
	OaU64 DmaRowPitch = 0;
	struct PendingDmaRelease {
		pw_buffer* Buffer = nullptr;
		OaImportedDmaBufImage Imported;
		OaCompletionToken Consumed;
	};
	OaVec<PendingDmaRelease> PendingDmaReleases;
#endif

	void FreeRing() {
		if (Engine == nullptr) return;
		for (auto& buffer : Ring) {
			if (buffer.Buffer != VK_NULL_HANDLE) {
				Engine->DeregisterBuffer(buffer);
				Engine->FreeBuffer(buffer);
			}
		}
		Ring.Clear();
		Head = 0;
		Latest = 0;
	}

	OaStatus EnsureRing(OaU32 InWidth, OaU32 InHeight) {
		if (Width == InWidth and Height == InHeight and Ring.Size() > 0U) {
			return OaStatus::Ok();
		}
		FreeRing();
		Width = InWidth;
		Height = InHeight;
		const OaU32 ringCount = std::max(2U, Config.RingFrames);
		const OaU64 bytes = static_cast<OaU64>(Width) * Height * 4ULL;
		Ring.Resize(ringCount);
		for (OaU32 index = 0; index < ringCount; ++index) {
			auto result = Engine->AllocBuffer(bytes);
			if (not result.IsOk()) {
				FreeRing();
				return result.GetStatus();
			}
			Ring[index] = OaStdMove(*result);
			Engine->RegisterBuffer(Ring[index]);
		}
		return OaStatus::Ok();
	}
};

#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
namespace {

VkFormat CaptureVkFormat(spa_video_format InFormat) {
	switch (InFormat) {
		case SPA_VIDEO_FORMAT_RGBA:
		case SPA_VIDEO_FORMAT_RGBx: return VK_FORMAT_R8G8B8A8_UNORM;
		case SPA_VIDEO_FORMAT_BGRA:
		case SPA_VIDEO_FORMAT_BGRx: return VK_FORMAT_B8G8R8A8_UNORM;
		default: return VK_FORMAT_UNDEFINED;
	}
}

const char* CaptureFormatName(spa_video_format InFormat) {
	switch (InFormat) {
		case SPA_VIDEO_FORMAT_RGBA: return "RGBA";
		case SPA_VIDEO_FORMAT_RGBx: return "RGBx";
		case SPA_VIDEO_FORMAT_BGRA: return "BGRA";
		case SPA_VIDEO_FORMAT_BGRx: return "BGRx";
		default: return "unknown";
	}
}

bool HasEnabledExtension(const OaVkDevice& InDevice, OaStringView InName) {
	for (const auto& extension : InDevice.Info.Software.EnabledDeviceExtensions) {
		if (extension == InName) return true;
	}
	return false;
}

bool IsImportableCaptureModifier(
	const OaComputeEngine& InEngine, VkFormat InFormat, OaU64 InModifier)
{
	VkPhysicalDeviceExternalImageFormatInfo externalInfo = {};
	externalInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
	externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
	VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifierInfo = {};
	modifierInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;
	modifierInfo.pNext = &externalInfo;
	modifierInfo.drmFormatModifier = InModifier;
	modifierInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkPhysicalDeviceImageFormatInfo2 formatInfo = {};
	formatInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
	formatInfo.pNext = &modifierInfo;
	formatInfo.format = InFormat;
	formatInfo.type = VK_IMAGE_TYPE_2D;
	formatInfo.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
	formatInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkExternalImageFormatProperties externalProperties = {};
	externalProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
	VkImageFormatProperties2 properties = {};
	properties.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
	properties.pNext = &externalProperties;
	const VkResult result = vkGetPhysicalDeviceImageFormatProperties2(
		static_cast<VkPhysicalDevice>(InEngine.Device.PhysicalDevice),
		&formatInfo, &properties);
	return result == VK_SUCCESS
		and (externalProperties.externalMemoryProperties.externalMemoryFeatures
			& VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0U;
}

std::vector<OaU64> CaptureDmaBufModifiers(
	const OaComputeEngine& InEngine, VkFormat InFormat)
{
	const auto& device = InEngine.Device;
	if (not HasEnabledExtension(device, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
		or not HasEnabledExtension(device, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)
		or not HasEnabledExtension(device, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
		return {};
	}
	VkDrmFormatModifierPropertiesListEXT list = {};
	list.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT;
	VkFormatProperties2 properties = {};
	properties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
	properties.pNext = &list;
	vkGetPhysicalDeviceFormatProperties2(
		static_cast<VkPhysicalDevice>(device.PhysicalDevice), InFormat, &properties);
	if (list.drmFormatModifierCount == 0U) return {};
	std::vector<VkDrmFormatModifierPropertiesEXT> candidates(list.drmFormatModifierCount);
	list.pDrmFormatModifierProperties = candidates.data();
	vkGetPhysicalDeviceFormatProperties2(
		static_cast<VkPhysicalDevice>(device.PhysicalDevice), InFormat, &properties);

	std::vector<OaU64> result;
	result.reserve(candidates.size());
	for (const auto& candidate : candidates) {
		if (candidate.drmFormatModifierPlaneCount != 1U
			or (candidate.drmFormatModifierTilingFeatures
				& VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0U) {
			continue;
		}
		if (IsImportableCaptureModifier(
			InEngine, InFormat, candidate.drmFormatModifier)) {
			result.push_back(candidate.drmFormatModifier);
		}
	}
	return result;
}

const spa_pod* AddCaptureFormat(
	spa_pod_builder& InBuilder,
	spa_video_format InFormat,
	const OaU64* InModifier,
	const spa_rectangle& InPreferred,
	const spa_rectangle& InMinimum,
	const spa_rectangle& InMaximum,
	const spa_fraction& InFps,
	const spa_fraction& InMinFps,
	const spa_fraction& InMaxFps)
{
	spa_pod_frame frame = {};
	spa_pod_builder_push_object(
		&InBuilder, &frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(&InBuilder,
		SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(InFormat),
		0);
	if (InModifier != nullptr) {
		spa_pod_builder_prop(
			&InBuilder, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY);
		spa_pod_builder_long(&InBuilder, static_cast<OaI64>(*InModifier));
	}
	spa_pod_builder_add(&InBuilder,
		SPA_FORMAT_VIDEO_size,
		SPA_POD_CHOICE_RANGE_Rectangle(&InPreferred, &InMinimum, &InMaximum),
		SPA_FORMAT_VIDEO_framerate,
		SPA_POD_CHOICE_RANGE_Fraction(&InFps, &InMinFps, &InMaxFps),
		0);
	return static_cast<const spa_pod*>(spa_pod_builder_pop(&InBuilder, &frame));
}

int QueueReleasedBuffer(spa_loop*, bool, OaU32, const void* InData,
	size_t InSize, void* InUserData)
{
	if (InSize != sizeof(pw_buffer*)) return -EINVAL;
	auto* stream = static_cast<pw_stream*>(InUserData);
	auto* buffer = *static_cast<pw_buffer* const*>(InData);
	return pw_stream_queue_buffer(stream, buffer);
}

void ReturnDmaBuffer(OaScreenCapture::Impl& InImpl, pw_buffer* InBuffer) {
	if (InBuffer == nullptr or InImpl.PwLoop == nullptr or InImpl.PwStream == nullptr) return;
	const int result = pw_loop_invoke(
		pw_main_loop_get_loop(InImpl.PwLoop), QueueReleasedBuffer, 0,
		&InBuffer, sizeof(InBuffer), true, InImpl.PwStream);
	if (result < 0) {
		OA_LOG_ERROR(OaLogComponent::App,
			"PipeWire screen capture could not return DMA-BUF: %s",
			std::strerror(-result));
	}
}

void PortalStartDone(GObject*, GAsyncResult* InResult, gpointer InData) {
	auto* impl = static_cast<OaScreenCapture::Impl*>(InData);
	if (not xdp_session_start_finish(impl->Session, InResult, &impl->PortalError)) {
		g_main_loop_quit(impl->PortalLoop);
		return;
	}
	GVariant* streams = xdp_session_get_streams(impl->Session);
	if (streams == nullptr) {
		impl->PortalError = g_error_new_literal(
			G_IO_ERROR, G_IO_ERROR_NOT_FOUND, "Portal returned no screencast streams");
		g_main_loop_quit(impl->PortalLoop);
		return;
	}
	GVariantIter iter;
	g_variant_iter_init(&iter, streams);
	GVariant* properties = nullptr;
	guint32 node = PW_ID_ANY;
	if (not g_variant_iter_next(&iter, "(u@a{sv})", &node, &properties)) {
		impl->PortalError = g_error_new_literal(
			G_IO_ERROR, G_IO_ERROR_NOT_FOUND, "Portal returned no screencast streams");
	} else {
		impl->PipeWireNode = node;
		g_variant_unref(properties);
		impl->PipeWireFd = xdp_session_open_pipewire_remote(impl->Session);
		if (impl->PipeWireFd < 0) {
			impl->PortalError = g_error_new_literal(
				G_IO_ERROR, G_IO_ERROR_FAILED, "Failed to open portal PipeWire remote");
		}
	}
	g_main_loop_quit(impl->PortalLoop);
}

void PortalSessionDone(GObject* InObject, GAsyncResult* InResult, gpointer InData) {
	auto* impl = static_cast<OaScreenCapture::Impl*>(InData);
	impl->Session = xdp_portal_create_screencast_session_finish(
		XDP_PORTAL(InObject), InResult, &impl->PortalError);
	if (impl->Session == nullptr) {
		g_main_loop_quit(impl->PortalLoop);
		return;
	}
	xdp_session_start(impl->Session, nullptr, nullptr, PortalStartDone, impl);
}

void StreamStateChanged(void* InData, pw_stream_state, pw_stream_state InState, const char* InError) {
	auto* impl = static_cast<OaScreenCapture::Impl*>(InData);
	if (InState == PW_STREAM_STATE_STREAMING) impl->Streaming = true;
	if (InState == PW_STREAM_STATE_ERROR or InState == PW_STREAM_STATE_UNCONNECTED) {
		impl->Streaming = false;
		if (InError != nullptr) {
			OA_LOG_ERROR(OaLogComponent::App, "PipeWire screen stream: %s", InError);
		}
	}
}

void StreamParamChanged(void* InData, OaU32 InId, const spa_pod* InParam) {
	auto* impl = static_cast<OaScreenCapture::Impl*>(InData);
	if (InId != SPA_PARAM_Format or InParam == nullptr) return;
	if (spa_format_video_raw_parse(InParam, &impl->RawFormat) < 0) return;
	switch (impl->RawFormat.format) {
		case SPA_VIDEO_FORMAT_RGBA:
		case SPA_VIDEO_FORMAT_RGBx:
		case SPA_VIDEO_FORMAT_BGRA:
		case SPA_VIDEO_FORMAT_BGRx:
			impl->SourceBytesPerPixel = 4U;
			break;
		default:
			pw_stream_set_error(impl->PwStream, -EINVAL,
				"OA screen capture negotiated an unsupported pixel format");
			return;
	}
	{
		std::lock_guard<std::mutex> lock(impl->FrameMutex);
		impl->CpuWidth = impl->RawFormat.size.width;
		impl->CpuHeight = impl->RawFormat.size.height;
		impl->CpuStride = impl->CpuWidth * 4U;
	}
	if ((impl->RawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U) {
		OA_LOG_INFO(OaLogComponent::App,
			"PipeWire screen capture: %ux%u %s via DMA-BUF modifier 0x%llx",
			impl->RawFormat.size.width, impl->RawFormat.size.height,
			CaptureFormatName(impl->RawFormat.format),
			static_cast<unsigned long long>(impl->RawFormat.modifier));
	} else {
		OA_LOG_INFO(OaLogComponent::App,
			"PipeWire screen capture: %ux%u %s via mapped memory",
			impl->RawFormat.size.width, impl->RawFormat.size.height,
			CaptureFormatName(impl->RawFormat.format));
	}

	OaU8 storage[512];
	spa_pod_builder builder = SPA_POD_BUILDER_INIT(storage, sizeof(storage));
	const spa_pod* params[1];
	const OaI32 dataTypes = (impl->RawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U
		? (1 << SPA_DATA_DmaBuf)
		: ((1 << SPA_DATA_MemPtr) | (1 << SPA_DATA_MemFd));
	params[0] = static_cast<const spa_pod*>(spa_pod_builder_add_object(
		&builder,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(4, 2, 8),
		SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
		SPA_PARAM_BUFFERS_size, SPA_POD_Int(static_cast<OaI32>(impl->RawFormat.size.width * impl->SourceBytesPerPixel * impl->RawFormat.size.height)),
		SPA_PARAM_BUFFERS_stride, SPA_POD_Int(static_cast<OaI32>(impl->RawFormat.size.width * impl->SourceBytesPerPixel)),
		SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(dataTypes)));
	pw_stream_update_params(impl->PwStream, params, 1);
}

void StreamProcess(void* InData) {
	auto* impl = static_cast<OaScreenCapture::Impl*>(InData);
	pw_buffer* pwBuffer = pw_stream_dequeue_buffer(impl->PwStream);
	if (pwBuffer == nullptr) return;
	spa_buffer* buffer = pwBuffer->buffer;
	if (buffer == nullptr or buffer->n_datas == 0U) {
		pw_stream_queue_buffer(impl->PwStream, pwBuffer);
		return;
	}
	spa_data& data = buffer->datas[0];
	if (data.chunk == nullptr or impl->CpuWidth == 0U or impl->CpuHeight == 0U) {
		pw_stream_queue_buffer(impl->PwStream, pwBuffer);
		return;
	}
	if (data.type == SPA_DATA_DmaBuf and data.fd >= 0
		and (impl->RawFormat.flags & SPA_VIDEO_FLAG_MODIFIER) != 0U) {
		std::lock_guard<std::mutex> lock(impl->FrameMutex);
		if (impl->HeldDmaBuffer != nullptr) {
			// One explicitly-held producer frame at a time. Returning additional
			// buffers keeps PipeWire live while the consumer finishes the held one.
			pw_stream_queue_buffer(impl->PwStream, pwBuffer);
			return;
		}
		impl->HeldDmaBuffer = pwBuffer;
		impl->DmaFd = static_cast<int>(data.fd);
		impl->DmaOffset = static_cast<OaU64>(data.mapoffset) + data.chunk->offset;
		impl->DmaRowPitch = data.chunk->stride > 0
			? static_cast<OaU64>(data.chunk->stride)
			: static_cast<OaU64>(impl->CpuWidth) * impl->SourceBytesPerPixel;
		impl->DmaTimestampUs = pw_stream_get_nsec(impl->PwStream) / 1000ULL;
		++impl->DmaSequence;
		return; // Release() queues this producer-owned buffer.
	}
	if (data.data == nullptr) {
		pw_stream_queue_buffer(impl->PwStream, pwBuffer);
		return;
	}
	const OaU8* source = static_cast<const OaU8*>(data.data) + data.chunk->offset;
	const OaU32 sourceStride = data.chunk->stride > 0
		? static_cast<OaU32>(data.chunk->stride)
		: impl->CpuWidth * impl->SourceBytesPerPixel;
	const OaU64 required = static_cast<OaU64>(impl->CpuWidth) * impl->CpuHeight * 4ULL;
	{
		std::lock_guard<std::mutex> lock(impl->FrameMutex);
		impl->CpuFrame.Resize(static_cast<OaUsize>(required));
		const bool swapRedBlue = impl->RawFormat.format == SPA_VIDEO_FORMAT_BGRA
			or impl->RawFormat.format == SPA_VIDEO_FORMAT_BGRx;
		const bool forceAlpha = impl->RawFormat.format == SPA_VIDEO_FORMAT_RGBx
			or impl->RawFormat.format == SPA_VIDEO_FORMAT_BGRx;
		for (OaU32 y = 0; y < impl->CpuHeight; ++y) {
			OaU8* destination = impl->CpuFrame.Data() + static_cast<OaU64>(y) * impl->CpuWidth * 4ULL;
			const OaU8* row = source + static_cast<OaU64>(y) * sourceStride;
			if (not swapRedBlue and not forceAlpha) {
				std::memcpy(destination, row, static_cast<OaUsize>(impl->CpuWidth) * 4U);
				continue;
			}
			for (OaU32 x = 0; x < impl->CpuWidth; ++x) {
				const OaU8* pixel = row + x * 4U;
				OaU8* out = destination + x * 4U;
				out[0] = swapRedBlue ? pixel[2] : pixel[0];
				out[1] = pixel[1];
				out[2] = swapRedBlue ? pixel[0] : pixel[2];
				out[3] = forceAlpha ? 255U : pixel[3];
			}
		}
		impl->CpuTimestampUs = pw_stream_get_nsec(impl->PwStream) / 1000ULL;
		++impl->CpuSequence;
	}
	pw_stream_queue_buffer(impl->PwStream, pwBuffer);
}

const pw_stream_events kStreamEvents = [] {
	pw_stream_events events = {};
	events.version = PW_VERSION_STREAM_EVENTS;
	events.state_changed = StreamStateChanged;
	events.param_changed = StreamParamChanged;
	events.process = StreamProcess;
	return events;
}();

} // namespace
#endif

OaScreenCapture::OaScreenCapture(OaScreenCapture&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_)) {}

OaScreenCapture& OaScreenCapture::operator=(OaScreenCapture&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}
OaScreenCapture::~OaScreenCapture() { Destroy(); }

bool OaScreenCapture::IsSupported() noexcept {
#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	return true;
#else
	return false;
#endif
}

OaResult<OaScreenCapture> OaScreenCapture::Open(
	OaComputeEngine& InEngine,
	const OaScreenCaptureConfig& InConfig)
{
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)InEngine;
	(void)InConfig;
	return OaStatus::Error(OaStatusCode::Unavailable,
		"Wayland screen capture requires libportal and PipeWire");
#else
	OaScreenCapture capture;
	capture.Impl_ = OaStdMakeUnique<Impl>();
	auto& impl = *capture.Impl_;
	impl.Engine = &InEngine;
	impl.Config = InConfig;
	impl.Portal = xdp_portal_new();
	impl.PortalLoop = g_main_loop_new(nullptr, FALSE);
	if (impl.Portal == nullptr or impl.PortalLoop == nullptr) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Could not initialize the desktop screencast portal");
	}

	XdpOutputType output = static_cast<XdpOutputType>(XDP_OUTPUT_MONITOR | XDP_OUTPUT_WINDOW);
	if (InConfig.Target == OaScreenCaptureTarget::Monitor) output = XDP_OUTPUT_MONITOR;
	if (InConfig.Target == OaScreenCaptureTarget::Window) output = XDP_OUTPUT_WINDOW;
	const XdpCursorMode cursor = InConfig.Cursor == OaScreenCaptureCursor::Embedded
		? XDP_CURSOR_MODE_EMBEDDED : XDP_CURSOR_MODE_HIDDEN;
	xdp_portal_create_screencast_session(
		impl.Portal, output, XDP_SCREENCAST_FLAG_NONE, cursor,
		XDP_PERSIST_MODE_TRANSIENT, nullptr, nullptr, PortalSessionDone, &impl);
	g_main_loop_run(impl.PortalLoop);
	if (impl.PortalError != nullptr) {
		OaString message = impl.PortalError->message;
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::PermissionDenied, message);
	}

	pw_init(nullptr, nullptr);
	impl.PwLoop = pw_main_loop_new(nullptr);
	if (impl.PwLoop == nullptr) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Could not create the PipeWire event loop");
	}
	impl.PwContext = pw_context_new(pw_main_loop_get_loop(impl.PwLoop), nullptr, 0);
	if (impl.PwContext == nullptr) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Could not create the PipeWire context");
	}
	impl.PwCore = pw_context_connect_fd(impl.PwContext, impl.PipeWireFd, nullptr, 0);
	if (impl.PwCore == nullptr) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::ConnectionFailed,
			"Could not connect to the portal PipeWire remote");
	}
	impl.PipeWireFd = -1; // ownership transferred to pw_core
	impl.PwStream = pw_stream_new(
		impl.PwCore, "oa-screen-capture",
		pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Video",
			PW_KEY_MEDIA_CATEGORY, "Capture",
			PW_KEY_MEDIA_ROLE, "Screen",
			nullptr));
	if (impl.PwStream == nullptr) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::Unavailable,
			"Could not create the PipeWire screen stream");
	}
	pw_stream_add_listener(impl.PwStream, &impl.StreamListener, &kStreamEvents, &impl);

	const spa_rectangle preferred = SPA_RECTANGLE(InConfig.PreferredWidth, InConfig.PreferredHeight);
	const spa_rectangle minimum = SPA_RECTANGLE(1, 1);
	const spa_rectangle maximum = SPA_RECTANGLE(16384, 16384);
	const spa_fraction fps = SPA_FRACTION(InConfig.PreferredFps, 1);
	const spa_fraction minFps = SPA_FRACTION(0, 1);
	const spa_fraction maxFps = SPA_FRACTION(240, 1);
	struct CaptureFormat {
		spa_video_format Spa;
		VkFormat Vk;
	};
	constexpr CaptureFormat formats[] = {
		{ SPA_VIDEO_FORMAT_RGBA, VK_FORMAT_R8G8B8A8_UNORM },
		{ SPA_VIDEO_FORMAT_RGBx, VK_FORMAT_R8G8B8A8_UNORM },
		{ SPA_VIDEO_FORMAT_BGRA, VK_FORMAT_B8G8R8A8_UNORM },
		{ SPA_VIDEO_FORMAT_BGRx, VK_FORMAT_B8G8R8A8_UNORM },
	};
	std::vector<std::pair<spa_video_format, OaU64>> dmaFormats;
	for (const auto& format : formats) {
		for (const OaU64 modifier : CaptureDmaBufModifiers(InEngine, format.Vk)) {
			dmaFormats.emplace_back(format.Spa, modifier);
		}
	}
	OA_LOG_INFO(OaLogComponent::App,
		"PipeWire screen capture: advertising %zu DMA-BUF format/modifier pairs and mapped fallback",
		dmaFormats.size());
	// Each pod is small, but use dynamically-sized stable storage so every
	// importable modifier can be offered without a fixed stack limit.
	std::vector<OaU8> storage(2048U + (dmaFormats.size() + 4U) * 512U);
	spa_pod_builder builder = SPA_POD_BUILDER_INIT(
		storage.data(), static_cast<OaU32>(storage.size()));
	std::vector<const spa_pod*> params;
	params.reserve(dmaFormats.size() + 4U);
	for (const auto& [format, modifier] : dmaFormats) {
		params.push_back(AddCaptureFormat(
			builder, format, &modifier, preferred, minimum, maximum,
			fps, minFps, maxFps));
	}
	// Mapped-memory formats are the universal fallback when the compositor
	// cannot allocate a modifier shared with the Vulkan device.
	for (const auto& format : formats) {
		params.push_back(AddCaptureFormat(
			builder, format.Spa, nullptr, preferred, minimum, maximum,
			fps, minFps, maxFps));
	}
	const int result = pw_stream_connect(
		impl.PwStream, PW_DIRECTION_INPUT, impl.PipeWireNode,
		static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS),
		params.data(), static_cast<OaU32>(params.size()));
	if (result < 0) {
		capture.Destroy();
		return OaStatus::Error(OaStatusCode::ConnectionFailed,
			"Could not connect the PipeWire screen stream");
	}
	impl.PwThread = std::thread([loop = impl.PwLoop] { pw_main_loop_run(loop); });
	return OaResult<OaScreenCapture>(OaStdMove(capture));
#endif
}

bool OaScreenCapture::Poll(OaVideoFrame& OutFrame) {
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)OutFrame;
	return false;
#else
	if (not Impl_) return false;
	auto& impl = *Impl_;
	// Retire producer buffers whose GPU consumers have completed. This keeps
	// the capture and encode queues fully asynchronous without letting
	// PipeWire overwrite a still-sampled DMA-BUF.
	for (OaUsize index = 0U; index < impl.PendingDmaReleases.Size();) {
		auto& pending = impl.PendingDmaReleases[index];
		if (not pending.Consumed.IsComplete()) { ++index; continue; }
		pending.Imported.Destroy();
		ReturnDmaBuffer(impl, pending.Buffer);
		impl.PendingDmaReleases.Erase(impl.PendingDmaReleases.Data() + index);
	}
	pw_buffer* rejectedDmaBuffer = nullptr;
	{
		std::lock_guard<std::mutex> lock(impl.FrameMutex);
		if (impl.HeldDmaBuffer != nullptr
			and impl.DmaSequence != impl.UploadedDmaSequence) {
			OaDmaBufImageDesc description;
			description.Fd = impl.DmaFd;
			description.Width = impl.CpuWidth;
			description.Height = impl.CpuHeight;
			description.Format = CaptureVkFormat(impl.RawFormat.format);
			description.Modifier = impl.RawFormat.modifier;
			description.Offset = impl.DmaOffset;
			description.RowPitch = impl.DmaRowPitch;
			auto imported = OaImportedDmaBufImage::Import(*impl.Engine, description);
			if (imported.IsOk()) {
				impl.ImportedDmaImage = OaStdMove(*imported);
				impl.UploadedDmaSequence = impl.DmaSequence;
				impl.Width = impl.CpuWidth;
				impl.Height = impl.CpuHeight;

				OutFrame = {};
				OutFrame.Resource = OaVideoFrameResource::Image;
				OutFrame.Image = impl.ImportedDmaImage.Image();
				OutFrame.ImageView = impl.ImportedDmaImage.View();
				OutFrame.Layout = VK_IMAGE_LAYOUT_GENERAL;
				OutFrame.ExternalQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
				OutFrame.Format = impl.ImportedDmaImage.Format();
				OutFrame.Width = impl.ImportedDmaImage.Width();
				OutFrame.Height = impl.ImportedDmaImage.Height();
				OutFrame.PresentationTimestamp = impl.DmaTimestampUs;
				OutFrame.Duration = impl.Config.PreferredFps > 0U
					? 1'000'000ULL / impl.Config.PreferredFps : 0ULL;
				OutFrame.IsRgb = true;
				OutFrame.ColorSpace = OaYCbCrModel::BT709;
				OutFrame.FullRange = true;
				return true;
			}
			OA_LOG_ERROR(OaLogComponent::App,
				"PipeWire supplied an advertised DMA-BUF that Vulkan could not import: %s",
				imported.GetStatus().ToString().c_str());
			rejectedDmaBuffer = impl.HeldDmaBuffer;
			impl.HeldDmaBuffer = nullptr;
			impl.DmaFd = -1;
			impl.DmaOffset = 0;
			impl.DmaRowPitch = 0;
			impl.UploadedDmaSequence = impl.DmaSequence;
		}
	}
	if (rejectedDmaBuffer != nullptr) {
		ReturnDmaBuffer(impl, rejectedDmaBuffer);
		return false;
	}

	std::lock_guard<std::mutex> lock(impl.FrameMutex);
	if (impl.CpuSequence == 0U or impl.CpuSequence == impl.UploadedCpuSequence) return false;
	if (not impl.EnsureRing(impl.CpuWidth, impl.CpuHeight).IsOk()) return false;
	auto& destination = impl.Ring[impl.Head];
	if (destination.MappedPtr == nullptr) return false;
	std::memcpy(destination.MappedPtr, impl.CpuFrame.Data(), impl.CpuFrame.Size());
	impl.Latest = impl.Head;
	impl.Head = (impl.Head + 1U) % static_cast<OaU32>(impl.Ring.Size());
	impl.UploadedCpuSequence = impl.CpuSequence;

	OutFrame = {};
	OutFrame.Resource = OaVideoFrameResource::Buffer;
	OutFrame.Buffer = &impl.Ring[impl.Latest];
	OutFrame.Format = VK_FORMAT_R8G8B8A8_UNORM;
	OutFrame.Width = impl.Width;
	OutFrame.Height = impl.Height;
	OutFrame.PresentationTimestamp = impl.CpuTimestampUs;
	OutFrame.Duration = impl.Config.PreferredFps > 0U
		? 1'000'000ULL / impl.Config.PreferredFps : 0ULL;
	OutFrame.IsRgb = true;
	OutFrame.ColorSpace = OaYCbCrModel::BT709;
	OutFrame.FullRange = true;
	return true;
#endif
}

void OaScreenCapture::Release(const OaVideoFrame& InFrame) {
	Release(InFrame, {});
}

void OaScreenCapture::Release(
	const OaVideoFrame& InFrame,
	const OaCompletionToken& InConsumed)
{
#if not defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	(void)InFrame;
	(void)InConsumed;
#else
	if (not Impl_ or InFrame.Resource != OaVideoFrameResource::Image) return;
	auto& impl = *Impl_;
	pw_buffer* buffer = nullptr;
	OaImportedDmaBufImage imported;
	{
		std::lock_guard<std::mutex> lock(impl.FrameMutex);
		if (not impl.ImportedDmaImage.IsValid()
			or InFrame.Image != impl.ImportedDmaImage.Image()) return;
		imported = OaStdMove(impl.ImportedDmaImage);
		buffer = impl.HeldDmaBuffer;
		impl.HeldDmaBuffer = nullptr;
		impl.DmaFd = -1;
		impl.DmaOffset = 0;
		impl.DmaRowPitch = 0;
	}
	if (InConsumed.IsValid() and not InConsumed.IsComplete()) {
		Impl::PendingDmaRelease pending;
		pending.Buffer = buffer;
		pending.Imported = OaStdMove(imported);
		pending.Consumed = InConsumed;
		impl.PendingDmaReleases.PushBack(OaStdMove(pending));
	} else {
		// Destroy the import only after the final GPU consumer returned the
		// image to FOREIGN ownership, then recycle the producer buffer.
		imported.Destroy();
		ReturnDmaBuffer(impl, buffer);
	}
#endif
}

void OaScreenCapture::Destroy() {
	if (not Impl_) return;
	auto& impl = *Impl_;
#if defined(OA_HAS_PIPEWIRE_SCREEN_CAPTURE)
	impl.Streaming = false;
	pw_buffer* heldDmaBuffer = nullptr;
	OaImportedDmaBufImage importedDmaImage;
	{
		std::lock_guard<std::mutex> lock(impl.FrameMutex);
		importedDmaImage = OaStdMove(impl.ImportedDmaImage);
		heldDmaBuffer = impl.HeldDmaBuffer;
		impl.HeldDmaBuffer = nullptr;
	}
	importedDmaImage.Destroy();
	ReturnDmaBuffer(impl, heldDmaBuffer);
	for (auto& pending : impl.PendingDmaReleases) {
		(void)pending.Consumed.Wait();
		pending.Imported.Destroy();
		ReturnDmaBuffer(impl, pending.Buffer);
	}
	impl.PendingDmaReleases.Clear();
	if (impl.PwLoop != nullptr) pw_main_loop_quit(impl.PwLoop);
	if (impl.PwThread.joinable()) impl.PwThread.join();
	if (impl.PwStream != nullptr) pw_stream_destroy(impl.PwStream);
	if (impl.PwCore != nullptr) pw_core_disconnect(impl.PwCore);
	if (impl.PwContext != nullptr) pw_context_destroy(impl.PwContext);
	if (impl.PwLoop != nullptr) pw_main_loop_destroy(impl.PwLoop);
	if (impl.Session != nullptr) {
		xdp_session_close(impl.Session);
		g_object_unref(impl.Session);
	}
	if (impl.Portal != nullptr) g_object_unref(impl.Portal);
	if (impl.PortalLoop != nullptr) g_main_loop_unref(impl.PortalLoop);
	if (impl.PortalError != nullptr) g_error_free(impl.PortalError);
#if defined(__linux__)
	if (impl.PipeWireFd >= 0) close(impl.PipeWireFd);
#endif
#endif
	impl.FreeRing();
	Impl_.reset();
}

bool OaScreenCapture::IsStreaming() const noexcept {
	return Impl_ and Impl_->Streaming.load();
}

OaU32 OaScreenCapture::Width() const noexcept { return Impl_ ? Impl_->Width : 0U; }
OaU32 OaScreenCapture::Height() const noexcept { return Impl_ ? Impl_->Height : 0U; }
