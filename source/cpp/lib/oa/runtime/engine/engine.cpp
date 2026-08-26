#include <oa/runtime/engine.h>
#include "engineAccess.h"
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/init.h>
#include <oa/runtime/uploadRing.h>
#include "../uploadRingRetirement.h"
#include "../presenterRetirement.h"
#include <oa/runtime/oaVk.h>
#include <oa/runtime/dispatch.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/dnn.h>
#include <oa/runtime/spirv.h>
#include <oa/runtime/matmulTypes.h>
#include "../gemm/engineRouteCacheAccess.h"
#include "../descriptorValidation.h"
#include "queueSubmitRoute.h"
#include <oa/core/envFlag.h>
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/matrix.h>
#include <oa/core/memory.h>
#include <oa/core/paths.h>
#include <oa/core/validation.h>
#include <oa/core/log.h>
#include <oa/core/device.h>
#include <oa/core/std/chrono.h>
#include <oa/core/thread.h>

// ─── Engine-local limits ─────────────────────────────────────────────────────────

static constexpr oa::U64 kHostVisibleCacheMaxBytes = 256ull * 1024ull * 1024ull;
static constexpr oa::U32 kHostVisibleCacheMaxBuffers = 4096u;
static bool asciiEqualsIgnoreCase(const char* inA, const char* inB);

oa::String oa::EngineConfig::defaultPipelineCacheDir_() {
	return oa::Paths::var("vk").string();
}

// Public matrices may outlive their engine, but their vulkan allocations may
// not outlive VMA. The registry owns that lifetime boundary: ordinary shared
// pointer release returns storage while the engine is live; close() drains all
// remaining allocations and leaves retained wrappers inert before destroying
// bindless/VMA/device state. The registry object itself is kept alive by those
// wrappers, so their eventual deleters never need to dereference oa::Engine.
class oa::Engine::Impl::BufferLeaseRegistry {
public:
	void attach(oa::Engine& inOwner) {
		oa::ScopedLock lock(mutex_);
		owner_ = &inOwner;
	}

	[[nodiscard]] bool registerBufferLease(oa::Engine& inOwner, oavk::Buffer* inBuffer) {
		if (not inBuffer) return false;
		oa::ScopedLock lock(mutex_);
		if (owner_ != &inOwner) return false;
		entries_.pushBack(inBuffer);
		return true;
	}

	void release(oavk::Buffer* inBuffer) {
		if (not inBuffer) return;
		{
			oa::ScopedLock lock(mutex_);
			for (oa::Usize i = 0; i < entries_.size(); ++i) {
				if (entries_[i] != inBuffer) continue;
				if (owner_) releaseStorage_(*owner_, *inBuffer, false);
				entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(i));
				break;
			}
		}
		delete inBuffer;
	}

	void drain(oa::Engine& inOwner) {
		oa::ScopedLock lock(mutex_);
		if (owner_ != &inOwner) return;
		// alias views are registered after their physical backing. Reverse order
		// destroys every alias VkBuffer before its VMA allocation.
		for (oa::Usize i = entries_.size(); i > 0; --i) {
			auto* buffer = entries_[i - 1];
			if (buffer) releaseStorage_(inOwner, *buffer, true);
		}
		entries_.clear();
		owner_ = nullptr;
	}

private:
	static void releaseStorage_(
		oa::Engine& inOwner, oavk::Buffer& inBuffer, oa::Bool inClosing)
	{
		if (not inBuffer.buffer) {
			inBuffer = {};
			return;
		}
		if (inClosing) {
			// Close must release the allocation now, not move it into the normal
			// host-visible reuse cache that is drained later in shutdown.
			inBuffer.flags |= OA_VK_BUFFER_FLAG_TRANSIENT;
		}
		oa::EngineAccess(inOwner).freeBuffer(inBuffer);
		inBuffer = {};
	}

	oa::Engine* owner_ = nullptr;
	oa::Vec<oavk::Buffer*> entries_;
	oa::Mutex mutex_;
};

oa::Engine::Impl::Impl() = default;
oa::Engine::Impl::~Impl() = default;

oa::Engine::Engine()
	: impl_(oa::makeUnique<Impl>())
{
	impl_->bufferLeaseRegistry_ = oa::makeShared<Impl::BufferLeaseRegistry>();
	impl_->bufferLeaseRegistry_->attach(*this);
}

static oa::MemoryPlacement resolveMatrixPlacement(const oavk::Device& inDevice) {
	const oa::String requested = oa::EnvFlag::getString("OA_MATRIX_MEMORY", "auto");
	if (asciiEqualsIgnoreCase(requested.cStr(), "device") || asciiEqualsIgnoreCase(requested.cStr(), "device-local")
		|| asciiEqualsIgnoreCase(requested.cStr(), "vram")) {
		return oa::MemoryPlacement::DeviceLocal;
	}
	if (asciiEqualsIgnoreCase(requested.cStr(), "host") || asciiEqualsIgnoreCase(requested.cStr(), "upload")
		|| asciiEqualsIgnoreCase(requested.cStr(), "mapped")) {
		return oa::MemoryPlacement::HostUpload;
	}
	if (asciiEqualsIgnoreCase(requested.cStr(), "unified") || asciiEqualsIgnoreCase(requested.cStr(), "bar")) {
		return oa::MemoryPlacement::Unified;
	}
	return inDevice.info.hardware.deviceType == oa::DeviceType::VkDiscrete
		? oa::MemoryPlacement::DeviceLocal
		: oa::MemoryPlacement::HostUpload;
}

void oa::EngineAccess::selectActiveSession() {
	oa::ExecutionSession::setActive(impl_->session_.get());
}

void oa::EngineAccess::clearActiveSession() {
	if (oa::ExecutionSession::getActivePtr() == impl_->session_.get()) {
		oa::ExecutionSession::setActive(nullptr);
	}
}

bool oa::Engine::hasCompute() const noexcept {
	return impl_->state_ == oa::EngineState::Ready;
}

bool oa::Engine::hasGraphics() const noexcept {
	return hasCompute()
		and impl_->device_.queues.graphicsQueueFamily != oavk::EnumerationIndexUnset;
}

bool oa::Engine::isReady() const noexcept {
	return impl_->state_ == oa::EngineState::Ready;
}

oa::EngineState oa::Engine::getState() const noexcept {
	return impl_->state_;
}

oa::U64 oa::Engine::deviceVramBytes() const noexcept {
	return impl_->device_.info.hardware.vramBytes;
}

oa::StringView oa::Engine::deviceName() const noexcept {
	return oa::StringView(impl_->device_.info.hardware.deviceName);
}

oa::StringView oa::Engine::deviceVendorName() const noexcept {
	return oa::StringView(impl_->device_.info.hardware.vendorName);
}

oa::DeviceType oa::Engine::deviceType() const noexcept {
	return impl_->device_.info.hardware.deviceType;
}

oa::StringView oa::Engine::driverName() const noexcept {
	return oa::StringView(impl_->device_.info.software.driverName);
}

oa::StringView oa::Engine::driverVersion() const noexcept {
	return oa::StringView(impl_->device_.info.software.driverVersion);
}

oa::StringView oa::Engine::vulkanApiVersion() const noexcept {
	return oa::StringView(impl_->device_.info.software.apiVersion);
}

oa::Precision oa::Engine::getPrecision() const noexcept {
	return impl_->precision_;
}

oa::MemoryUsage oa::Engine::getMemoryUsage() const {
	if (impl_->state_ != oa::EngineState::Ready) return {};
	const auto stats = impl_->allocator_.getStats();
	const oa::U64 total = stats.budgetBytes != 0U
		? stats.budgetBytes
		: impl_->device_.info.hardware.vramBytes;
	if (total == 0U) return {};
	const oa::U64 used = oa::min(stats.usedBytes, total);
	return oa::MemoryUsage{
		.totalBytes = total,
		.freeBytes = total - used,
		.usedBytes = used,
		.usedPercent = 100.0 * static_cast<oa::F64>(used)
			/ static_cast<oa::F64>(total),
	};
}

oa::Result<oa::ExecutionPlan> oa::Engine::capture(oa::Fn<void()> inRecord) {
	if (impl_->state_ != oa::EngineState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::capture requires a ready engine");
	}
	if (not inRecord) {
		return oa::Status::invalidArgument(
			"oa::Engine::capture requires a recording callback");
	}

	auto captureSession = oa::makeUnique<oa::ExecutionSession>(this);
	{
		oa::ExecutionSession::RecordingScope recording(*captureSession);
		// Recording callbacks obey OA's no-exception contract. Operation failures
		// are retained by the session and reported by finalize/submit below.
		inRecord();
	}

	auto validation = captureSession->validateCapture();
	if (not validation.isOk()) {
		captureSession->clear();
		return validation;
	}

	const auto* source = captureSession->graph();
	const auto* semanticSource = captureSession->semanticGraph();
	if (source == nullptr or semanticSource == nullptr) {
		captureSession->clear();
		return oa::Status::error(oa::StatusCode::Internal,
			"oa::Engine::capture has no coherent recording");
	}
	auto dnnResult = oa::DnnPlanner::plan(*semanticSource);
	if (not dnnResult.isOk()) {
		captureSession->clear();
		return dnnResult.getStatus();
	}

	oa::ExecutionPlan plan;
	plan.dnnPlan_ = oa::makeUnique<oa::DnnPlan>(
		oa::move(dnnResult).getValue());
	auto copyStatus = plan.graph().copyNodesFrom(*source);
	if (not copyStatus.isOk()) {
		captureSession->clear();
		return copyStatus;
	}
	auto compileStatus = plan.compile(*this);
	if (not compileStatus.isOk()) {
		captureSession->clear();
		return compileStatus;
	}

	// Compilation retained every resource referenced by the copied nodes. clear
	// the temporary authoring session without submitting its source recording.
	captureSession->clear();
	return oa::Result<oa::ExecutionPlan>(oa::move(plan));
}

oa::Result<oa::Event> oa::Engine::submit(oa::ExecutionPlan& inPlan) {
	if (impl_->state_ != oa::EngineState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::submit requires a ready engine");
	}
	if (not inPlan.isCompiled() or inPlan.engine_ == nullptr) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::submit requires a compiled execution plan");
	}
	if (inPlan.engine_ != this) {
		return oa::Status::invalidArgument(
			"oa::Engine::submit cannot submit a plan captured by another engine");
	}
	return inPlan.replayAsync();
}

oa::Result<oa::Event> oa::Engine::submit(oa::Timer* inTimer) {
	if (impl_->state_ != oa::EngineState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::submit requires a ready engine");
	}
	return oa::ExecutionSession::forEngine(*this).submit(inTimer);
}

oa::Status oa::Engine::wait(const oa::Event& inEvent) {
	if (impl_->state_ != oa::EngineState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::wait requires a ready engine");
	}
	if (not ownsEvent(inEvent)) {
		return oa::Status::invalidArgument(
			"oa::Engine::wait cannot wait for an event from another engine");
	}
	auto& context = oa::ExecutionSession::forEngine(*this);
	if (context.isPendingEvent(inEvent)) {
		return context.wait(inEvent);
	}
	return inEvent.wait();
}

const oa::ExecutionStats& oa::Engine::lastExecutionStats() const noexcept {
	return oa::ExecutionSession::forEngine(*this).lastExecutionStats();
}

bool oa::Engine::supportsClockCalibration() const noexcept {
	return impl_->state_ == oa::EngineState::Ready
		and impl_->device_.info.hardware.computeTimestampValidBits != 0U
		and (impl_->device_.info.software.hasKhrCalibratedTimestamps
			or impl_->device_.info.software.hasExtCalibratedTimestamps);
}

oa::Result<oa::ClockCalibration> oa::Engine::calibrateClock(
	oa::U32 inSampleCount) const
{
	if (impl_->state_ != oa::EngineState::Ready) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::calibrateClock requires a ready engine");
	}
	if (inSampleCount == 0U or inSampleCount > 1024U) {
		return oa::Status::invalidArgument(
			"oa::Engine::calibrateClock sample count must be in [1, 1024]");
	}
	if (not supportsClockCalibration()) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"selected device does not expose calibrated timestamps");
	}

	using GetTimeDomainsFn = VkResult (VKAPI_PTR *)(
		VkPhysicalDevice, oa::U32*, VkTimeDomainKHR*);
	using GetCalibratedTimestampsFn = VkResult (VKAPI_PTR *)(
		VkDevice, oa::U32, const VkCalibratedTimestampInfoKHR*, oa::U64*, oa::U64*);

	const auto instance = static_cast<VkInstance>(impl_->device_.instance);
	const auto physical = static_cast<VkPhysicalDevice>(impl_->device_.physicalDevice);
	const auto device = static_cast<VkDevice>(impl_->device_.device);
	const bool useKhr = impl_->device_.info.software.hasKhrCalibratedTimestamps;
	const char* domainName = useKhr
		? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR"
		: "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT";
	const char* calibrationName = useKhr
		? "vkGetCalibratedTimestampsKHR"
		: "vkGetCalibratedTimestampsEXT";
	const auto getInstanceProcAddr = oaVkGetInstanceProcAddr();
	const auto getDomains = reinterpret_cast<GetTimeDomainsFn>(
		getInstanceProcAddr != nullptr
			? getInstanceProcAddr(instance, domainName)
			: nullptr);
	const auto getCalibrated = reinterpret_cast<GetCalibratedTimestampsFn>(
		impl_->device_.instanceDispatch.vkGetDeviceProcAddr(device, calibrationName));
	if (getDomains == nullptr or getCalibrated == nullptr) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"calibrated timestamp entry points are unavailable");
	}

	oa::U32 domainCount = 0U;
	VkResult result = getDomains(physical, &domainCount, nullptr);
	if (result != VK_SUCCESS or domainCount == 0U) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"calibrated timestamp domain query failed");
	}
	oa::Vec<VkTimeDomainKHR> domains(domainCount);
	result = getDomains(physical, &domainCount, domains.data());
	if (result != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"calibrated timestamp domain enumeration failed");
	}

	bool hasDevice = false;
	bool hasMonotonic = false;
	bool hasMonotonicRaw = false;
	for (oa::U32 i = 0; i < domainCount; ++i) {
		hasDevice = hasDevice or domains[i] == VK_TIME_DOMAIN_DEVICE_KHR;
		hasMonotonic = hasMonotonic
			or domains[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
		hasMonotonicRaw = hasMonotonicRaw
			or domains[i] == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
	}
	if (not hasDevice or (not hasMonotonic and not hasMonotonicRaw)) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"device and POSIX monotonic clock domains cannot be calibrated together");
	}

	const VkTimeDomainKHR hostDomain = hasMonotonicRaw
		? VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR
		: VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
	const VkCalibratedTimestampInfoKHR infos[2] = {
		{
			.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
			.pNext = nullptr,
			.timeDomain = VK_TIME_DOMAIN_DEVICE_KHR,
		},
		{
			.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
			.pNext = nullptr,
			.timeDomain = hostDomain,
		},
	};

	oa::ClockCalibration best{};
	bool haveBest = false;
	for (oa::U32 sample = 0; sample < inSampleCount; ++sample) {
		oa::U64 timestamps[2]{};
		oa::U64 maximumDeviation = 0U;
		result = getCalibrated(device, 2U, infos, timestamps, &maximumDeviation);
		if (result != VK_SUCCESS) {
			return oa::Status::error(oa::StatusCode::VulkanError,
				"vkGetCalibratedTimestamps failed");
		}
		if (haveBest and maximumDeviation >= best.maximumDeviationNanoseconds) {
			continue;
		}

		const oa::U32 validBits =
			impl_->device_.info.hardware.computeTimestampValidBits;
		if (validBits < 64U) {
			timestamps[0] &= (oa::U64{1} << validBits) - 1U;
		}
		best.deviceTimestampTicks = timestamps[0];
		best.hostTimestampNanoseconds = timestamps[1];
		best.maximumDeviationNanoseconds = maximumDeviation;
		best.deviceNanosecondsPerTick =
			impl_->device_.info.hardware.timestampPeriodNanoseconds;
		best.deviceTimestampValidBits = validBits;
		best.hostClockDomain = hasMonotonicRaw
			? oa::HostClockDomain::MonotonicRaw
			: oa::HostClockDomain::Monotonic;
		haveBest = true;
	}
	return best;
}

static oa::U32 enumerationIndexOrZero(const oavk::Device& inDev) {
	return inDev.info.hardware.enumerationIndex != oavk::EnumerationIndexUnset
		? inDev.info.hardware.enumerationIndex
		: 0u;
}

enum class DeviceInitLogMode : oa::U8 {
	Compact,
	Full,
	Off,
};

static bool asciiEqualsIgnoreCase(const char* inA, const char* inB) {
	if (inA == nullptr || inB == nullptr) {
		return inA == inB;
	}
	while (*inA != '\0' && *inB != '\0') {
		const auto lowerAscii = [](char inValue) noexcept {
			return inValue >= 'A' && inValue <= 'Z'
				? static_cast<char>(inValue + ('a' - 'A'))
				: inValue;
		};
		const char a = lowerAscii(*inA);
		const char b = lowerAscii(*inB);
		if (a != b) {
			return false;
		}
		++inA;
		++inB;
	}
	return *inA == '\0' && *inB == '\0';
}

static DeviceInitLogMode getDeviceInitLogMode() {
	const oa::String env = oa::EnvFlag::getString("OA_LOG_DEVICE_INIT", "");
	if (!env.empty()) {
		const char* value = env.cStr();
		if (asciiEqualsIgnoreCase(value, "full") ||
		    asciiEqualsIgnoreCase(value, "debug") ||
		    asciiEqualsIgnoreCase(value, "verbose") ||
		    asciiEqualsIgnoreCase(value, "1") ||
		    asciiEqualsIgnoreCase(value, "true") ||
		    asciiEqualsIgnoreCase(value, "on")) {
			return DeviceInitLogMode::Full;
		}
		if (asciiEqualsIgnoreCase(value, "compact") ||
		    asciiEqualsIgnoreCase(value, "minimal") ||
		    asciiEqualsIgnoreCase(value, "min")) {
			return DeviceInitLogMode::Compact;
		}
		if (asciiEqualsIgnoreCase(value, "off") ||
		    asciiEqualsIgnoreCase(value, "none") ||
		    asciiEqualsIgnoreCase(value, "0") ||
		    asciiEqualsIgnoreCase(value, "false") ||
		    asciiEqualsIgnoreCase(value, "no")) {
			return DeviceInitLogMode::Off;
		}
	}
#ifdef NDEBUG
	return DeviceInitLogMode::Compact;
#else
	return DeviceInitLogMode::Full;
#endif
}

static void logSelectedDevicesSummarySingle(const oavk::Device& inDev) {
	OaLogInfo(oa::LogComponent::Engine, "Selected devices: (");
	OaLogInfo(oa::LogComponent::Engine,
		"  (0) [Vk %u]: %s — %s",
		static_cast<unsigned>(enumerationIndexOrZero(inDev)),
		inDev.info.hardware.vendorName.cStr(),
		inDev.info.hardware.deviceName.cStr());
	OaLogInfo(oa::LogComponent::Engine, ")");
}

void oa::EngineAccess::logSelectedDevices() {
	const DeviceInitLogMode mode = getDeviceInitLogMode();
	if (mode == DeviceInitLogMode::Off) {
		return;
	}
	if (mode == DeviceInitLogMode::Compact) {
		impl_->device_.printInfoCompact();
		return;
	}

	logSelectedDevicesSummarySingle(impl_->device_);
	impl_->device_.printInfoDetailed();
	if (oa::EnvFlag::isSet("OA_LOG_COOPMAT_SHAPES")) {
		oavk::logCoopMatShapes(impl_->device_.info.software.coopMatShapes, "      ");
	}
}

oa::Engine::~Engine() {
	if (auto status = close(); !status.isOk()) {
		OaLogError(oa::LogComponent::Engine,
			"oa::Engine::~Engine: shutdown failed: %s",
			status.toString().cStr());
		// completion did not prove that retired service resources are idle. The
		// engine cannot report another retry once destruction begins, so detach
		// these payloads instead of invoking release callbacks against potentially
		// live vulkan work. This is an intentional last-resort leak.
		oa::EngineAccess(*this).detachRetiredBorrowedServices();
	}
}

oa::SharedPtr<oavk::Buffer> oa::EngineAccess::adoptBufferLease(
	oavk::Buffer&& inBuffer,
	oa::SharedPtr<oavk::Buffer> inBacking)
{
	auto* buffer = new oavk::Buffer(oa::move(inBuffer));
	auto registry = impl_->bufferLeaseRegistry_;
	if (not registry or not registry->registerBufferLease(engine_, buffer)) {
		freeBuffer(*buffer);
		delete buffer;
		return {};
	}
	return oa::SharedPtr<oavk::Buffer>(buffer,
		[registry = oa::move(registry), backing = oa::move(inBacking)](
			oavk::Buffer* inPtr) mutable {
			registry->release(inPtr);
			backing.reset();
		});
}

// oa::Engine is pinned (move/copy = delete in the header). It owns a
// VkInstance/VkDevice/VMA/queues/mutexes and self-referential pools, so it must
// never relocate. create() builds it on the heap and returns an owning pointer;
// there is deliberately no move ctor/assignment here. The old hand-written move
// surgery that shallow-copied vulkan/VMA ownership is gone.

oa::Result<oa::UniquePtr<oa::Engine>> oa::Engine::create(const oa::EngineConfig& inConfig) {
	// Pinned construction: build the engine on the heap so its address remains
	// stable for private contexts, pools, and retirement state. No build-then-move.
	oa::UniquePtr<oa::Engine> engine(new oa::Engine());
	const oa::Status initStatus = oa::EngineAccess(*engine).initialize(inConfig);
	if (not initStatus.isOk()) {
		const oa::Status closeStatus = engine->close();
		if (not closeStatus.isOk()) {
			oa::String message = "oa::Engine initialization failed: ";
			message += initStatus.toString();
			message += "; partial-engine cleanup failed: ";
			message += closeStatus.toString();
			return oa::Status::error(closeStatus.getCode(), oa::move(message));
		}
		return initStatus;
	}
	return engine;
}

oa::Status oa::EngineAccess::initialize(const oa::EngineConfig& inConfig) {
	if (impl_->state_ != oa::EngineState::Empty) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::Engine::create: engine initialization is one-shot");
	}
	impl_->state_ = oa::EngineState::Initializing;
	// Validation evidence is intentionally selectable at runtime so the exact
	// release workload can be exercised under core, synchronization, and
	// GPU-assisted VVL profiles. An explicit config request or the diagnostic
	// environment toggle enables it; a false environment value never disables
	// an explicit config request.
	oa::EngineConfig resolvedConfig = inConfig;
	resolvedConfig.enableValidation = inConfig.enableValidation
		or oa::EnvFlag::isSet("OA_VK_VALIDATION");
	auto status = initializeImpl(resolvedConfig);
	impl_->state_ = status.isOk() ? oa::EngineState::Ready : oa::EngineState::Failed;
	return status;
}

oa::Status oa::EngineAccess::initializeImpl(const oa::EngineConfig& inConfig) {
	oa::Engine& rt = engine_;
	auto logger = oa::Log::create(inConfig.log);
	if (not logger.isOk()) return logger.getStatus();
	rt.impl_->logger_ = oa::move(logger).getValue();
	rt.impl_->previousLogSelection_ = oa::LogAccess::select(rt.impl_->logger_.get());
	switch (inConfig.precision) {
		case oa::Precision::FP32:
		case oa::Precision::BF16:
			break;
		case oa::Precision::FP64:
			return oa::Status::error(oa::StatusCode::Unimplemented,
				"FP64 requires a complete Float64 kernel, autograd, optimizer, and "
				"numerical-validation pack; OA never substitutes FP32 for FP64");
		default:
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"unknown engine precision policy");
	}

	oa::Validation::initFromEnv();
	// translate oa::NumericMode → env-knob state BEFORE any subsystem reads env.
	// User-supplied env vars still win (SetIfUnset checks first).
	oa::applyNumericMode(inConfig.numericMode);

	oa::DeviceType pref = oa::DeviceType::VkDiscrete;
	switch (inConfig.devicePref) {
		case oa::DevicePreference::Integrated: pref = oa::DeviceType::VkIntegrated; break;
		case oa::DevicePreference::Cpu:        pref = oa::DeviceType::VkCpu; break;
		default: break;
	}

	oa::Vec<const char*> instanceExtraPtrs;
	for (const auto& ext : inConfig.instanceExtraExtensions) {
		if (!ext.empty()) {
			instanceExtraPtrs.pushBack(ext.cStr());
		}
	}
	oa::Span<const char* const> extraSpan(instanceExtraPtrs.data(), instanceExtraPtrs.size());

	// Graphics queue selection and WSI extension admission are independent.
	// Headless must not enable extensions whose registry dependency is
	// VK_KHR_surface.
	const oa::Bool wantPresentation =
		inConfig.presentationMode == oa::PresentationMode::Swapchain;
	const oa::Bool wantHeadlessGraphics =
		inConfig.presentationMode == oa::PresentationMode::Headless;

	auto pickResult = inConfig.devicePref == oa::DevicePreference::ByIndex
		? oavk::Device::create(
			oa::StringView(inConfig.appName),
			inConfig.enableValidation,
			pref,
			inConfig.deviceIndex,
			inConfig.appVersion,
			extraSpan,
			wantPresentation,
			wantHeadlessGraphics
		) : oavk::Device::create(
			oa::StringView(inConfig.appName),
			inConfig.enableValidation,
			pref,
			oavk::EnumerationIndexUnset,
			inConfig.appVersion,
			extraSpan,
			wantPresentation,
			wantHeadlessGraphics
		);

	if (!pickResult.isOk()) {
		return pickResult.getStatus();
	}

	auto allocator = OaVma::create(pickResult.getValue());
	if (!allocator.isOk()) {
		pickResult.getValue().destroy();
		return allocator.getStatus();
	}

	rt.impl_->device_ = oa::move(pickResult.getValue());
	rt.impl_->allocator_ = oa::move(allocator.getValue());
	rt.impl_->timerRegistry_ = oa::makeShared<oa::TimerRegistry>(rt);
	rt.impl_->precision_ = inConfig.precision;
	rt.impl_->matrixPlacement_ = resolveMatrixPlacement(rt.impl_->device_);

	if (rt.impl_->precision_ == oa::Precision::BF16 && !rt.impl_->device_.nativeShaderBfloat16Usable()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"BF16 requested but the selected device lacks a trusted native "
			"VK_KHR_shader_bfloat16 path; OA never silently substitutes FP32");
	}

	logSelectedDevices();
	if (inConfig.enableValidation) {
		OaLogInfo(oa::LogComponent::Engine, "Validation layers: ON (VK_LAYER_KHRONOS_validation)");
	}

	auto bindlessResult = oavk::BindlessHeap::create(rt.impl_->device_);
	if (!bindlessResult.isOk()) {
		return oa::Status::error(oa::StatusCode::PipelineError,
			"bindless heap creation failed (required for all operations)");
	}
	rt.impl_->bindless_ = oa::move(bindlessResult.getValue());
	rt.impl_->session_ = oa::makeUnique<oa::ExecutionSession>(&rt);

	oa::String cacheDir;
	if (inConfig.enablePipelineCache) {
		cacheDir = inConfig.pipelineCacheDir;
	}
	OA_RETURN_IF_ERROR(rt.impl_->pipelines_.init(
		rt.impl_->device_, cacheDir, rt.impl_->bindless_.pipelineLayout));

	if (inConfig.selectForThread) {
		selectActiveSession();
	}

	// Pre-load all embedded shaders from the registry
	if (inConfig.preloadEmbeddedPipelines) {
		auto loadStatus = ensureAllEmbeddedLiboaPipelines();
		if (!loadStatus.isOk()) {
			OaLogWarn(oa::LogComponent::Engine, "Failed to pre-load embedded pipelines: %s",
				loadStatus.getMessage().cStr());
		}
	}

	// initialize GEMM route cache
	rt.impl_->gemmState_ = oa::makeUnique<Impl::GemmState>();
	rt.impl_->gemmState_->routeCache = oa::makeUnique<oa::GemmRouteCache>();

	// load cached route policy from disk if available
	const char* cachePath = oa::GemmRouteCache::DefaultPath;
	(void)rt.impl_->gemmState_->routeCache->load(cachePath);

	return oa::Status::ok();
}

oa::Status oa::Engine::close() {
	oa::EngineAccess access(*this);
	// Guard against an explicit close() followed by the destructor. The engine is
	// pinned, so there is no moved-from state to guard.
	if (impl_->state_ == oa::EngineState::Destroying || impl_->state_ == oa::EngineState::Destroyed) {
		return oa::Status::ok();
	}
	oa::Status firstError = oa::Status::ok();
	auto retainError = [&firstError](const oa::Status& inStatus) {
		if (firstError.isOk() && !inStatus.isOk()) firstError = inStatus;
	};
	impl_->state_ = oa::EngineState::Destroying;
	if (!impl_->device_.device) {
		if (impl_->bufferLeaseRegistry_) impl_->bufferLeaseRegistry_->drain(*this);
		if (impl_->logger_) {
			oa::LogAccess::restoreIfCurrent(
				impl_->logger_.get(), impl_->previousLogSelection_);
			retainError(impl_->logger_->close());
			impl_->logger_.reset();
		}
		impl_->state_ = oa::EngineState::Destroyed;
		return firstError;
	}
	access.clearActiveSession();

	// Engine shutdown owns completion. finish engine-owned stream timelines while
	// context graphs, descriptor pools, and retained buffer owners are alive.
	// Abandoned recording is cancelled by its execution session; shutdown never
	// turns an unsubmitted recording into work.
	for (auto& stream : impl_->streamPool_) {
		if (stream && stream->submitted) retainError(stream->synchronize(impl_->device_));
	}
	for (auto& stream : impl_->asyncStreamPool_) {
		if (stream && stream->submitted) retainError(stream->synchronize(impl_->device_));
	}
	if (impl_->transferStream_.submitted) retainError(impl_->transferStream_.synchronize(impl_->device_));
	if (impl_->readbackStream_.submitted) retainError(impl_->readbackStream_.synchronize(impl_->device_));
	// Graphics leases abandoned by composed services retire into the engine pool.
	// Engine Close is the explicit boundary that completes those timelines and
	// cancels any never-submitted recording before borrowed service payloads can
	// release resources referenced by those submissions.
	const oa::Status graphicsStatus = access.completeGraphicsStreams();
	retainError(graphicsStatus);
	if (not graphicsStatus.isOk()) {
		impl_->state_ = oa::EngineState::Failed;
		return firstError;
	}
	// A failed borrowed-service completion retains its payload for a later Close
	// retry. Do not enter device teardown until every callback proves completion.
	const oa::Status borrowedServiceStatus = access.completeRetiredBorrowedServices();
	retainError(borrowedServiceStatus);
	if (not borrowedServiceStatus.isOk()) {
		impl_->state_ = oa::EngineState::Failed;
		return firstError;
	}
	if (impl_->uploadRing_) {
		retainError(impl_->uploadRing_->close());
		impl_->uploadRing_.reset();
	}
	// Directly pooled engine stream timelines are complete now. Retire facade-
	// owned submissions and their resources before either subsystem is destroyed.
	access.collectRetiredImageDispatches();
	retainError(access.completeRetiredExecutionPlans());
	retainError(access.completeRetiredSessionBatches());
	retainError(access.completeRetiredUploadRings());
	retainError(access.completeRetiredPresenters());

	impl_->session_.reset();
	// external matrix owners can outlive the engine object, but their allocations
	// cannot outlive its allocator. drain them after all GPU timelines/context
	// graphs complete and before bindless/VMA/device teardown. Retained wrappers
	// are zeroed, so later matrix inspection/destruction is inert.
	if (impl_->bufferLeaseRegistry_) impl_->bufferLeaseRegistry_->drain(*this);

	// Cleanup GEMM route cache
	if (impl_->gemmState_ and impl_->gemmState_->routeCache) {
		const char* cachePath = oa::GemmRouteCache::DefaultPath;
		(void)impl_->gemmState_->routeCache->save(cachePath);
	}
	impl_->gemmState_.reset();

	// destroy engine-level stream pools (always use primary device)
	for (auto& s : impl_->streamPool_) s->destroy(impl_->device_);
	impl_->streamPool_.clear();
	impl_->freeStack_.clear();
	for (auto& s : impl_->asyncStreamPool_) s->destroy(impl_->device_);
	impl_->asyncStreamPool_.clear();
	impl_->asyncFreeStack_.clear();
	for (auto& slot : impl_->graphicsStreamPool_) {
		if (slot.stream) slot.stream->destroy(impl_->device_);
	}
	impl_->graphicsStreamPool_.clear();
	impl_->transferStream_.destroy(impl_->device_);
	impl_->readbackStream_.destroy(impl_->device_);
	if (impl_->timerRegistry_) {
		retainError(impl_->timerRegistry_->close(impl_->device_));
		impl_->timerRegistry_.reset();
	}
	impl_->allocator_.free(impl_->readbackStaging_);

	{
		oa::ScopedLock lock(impl_->hostVisibleBufferCacheMutex_);
		for (auto& entry : impl_->hostVisibleBufferCache_) {
			access.deregisterBuffer(entry.buffer);
			impl_->allocator_.free(entry.buffer);
		}
		impl_->hostVisibleBufferCache_.clear();
		impl_->hostVisibleBufferCacheBytes_ = 0;
	}

	impl_->pipelines_.destroy(impl_->device_);
	impl_->bindless_.destroy(impl_->device_);
	impl_->allocator_.destroy();
	impl_->device_.destroy();
	if (impl_->logger_) {
		oa::LogAccess::restoreIfCurrent(
			impl_->logger_.get(), impl_->previousLogSelection_);
		retainError(impl_->logger_->close());
		impl_->logger_.reset();
	}
	impl_->state_ = oa::EngineState::Destroyed;
	return firstError;
}

void oa::EngineAccess::retireBorrowedService(
	void* inPayload,
	RetiredServiceCompleteFn inComplete,
	RetiredServiceReleaseFn inRelease)
{
	if (not inPayload) return;
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredBorrowedServiceMutex_);
	impl_->retiredBorrowedServices_.emplaceBack(
		inPayload, inComplete, inRelease);
}

oa::Status oa::EngineAccess::completeRetiredBorrowedServices() {
	oa::Vec<Impl::RetiredServiceState> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredBorrowedServiceMutex_);
		retired = oa::move(impl_->retiredBorrowedServices_);
	}
	oa::Status firstError = oa::Status::ok();
	oa::Vec<Impl::RetiredServiceState> retry;
	for (auto& service : retired) {
		const auto status = service.complete();
		if (not status.isOk()) {
			if (firstError.isOk()) firstError = status;
			retry.pushBack(oa::move(service));
		}
	}
	if (not retry.empty()) {
		// completion callbacks may retire additional services. append failures to
		// that live queue before the local vector destructs so their release
		// callbacks cannot run until a successful retry.
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredBorrowedServiceMutex_);
		impl_->retiredBorrowedServices_.reserve(
			impl_->retiredBorrowedServices_.size() + retry.size());
		for (auto& service : retry) {
			impl_->retiredBorrowedServices_.pushBack(oa::move(service));
		}
	}
	return firstError;
}

void oa::EngineAccess::detachRetiredBorrowedServices() noexcept {
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredBorrowedServiceMutex_);
	for (auto& service : impl_->retiredBorrowedServices_) {
		service.detachWithoutRelease();
	}
	impl_->retiredBorrowedServices_.clear();
}

oa::Status oa::EngineAccess::ensurePipeline(
	oa::StringView inName,
	oa::Span<const oa::U8> inSpirv,
	const oa::PipelineSpec& inSpec)
{
	return impl_->pipelines_.ensurePipeline(impl_->device_, inName, inSpirv, inSpec);
}

oa::Status oa::EngineAccess::ensureAllEmbeddedLiboaPipelines() {
	oa::U32 loaded = 0;
	oa::U32 skipped = 0;
	oa::U32 failed = 0;
	const oa::U32 total = oavk::spirvCount();

	class ShaderLoadPlan {
	public:
		const char* name = nullptr;
		oa::U32 firstRequest = 0;
		oa::U32 requestCount = 0;
	};
	oa::Vec<oa::PipelineLoadRequest> requests;
	oa::Vec<ShaderLoadPlan> plans;
	requests.reserve(total * (impl_->device_.nativeShaderBfloat16Usable() ? 2u : 1u));
	plans.reserve(total);

	for (oa::U32 idx = 0; idx < total; ++idx) {
		const oavk::SpirvEntry* ent = oavk::findSpirvByIndex(idx);
		if (!ent || !ent->name || ent->size == 0 || !ent->data) continue;

		if (!oa::computeKernelUsesDefaultBindlessPipeline(ent->name)) {
			++skipped;
			continue;
		}

		// Feature-gate exact matmul variants from their registry metadata. The
		// device mask is already vendor-trust-corrected, so unsupported variants
		// are excluded before vulkan pipeline creation and routing falls back to a
		// legal implementation. Non-matmul shaders are capability-neutral here.
		if (const auto* variant = oa::matmulRegistry::findByShaderName(ent->name);
			variant != nullptr
			and !oa::matmulRegistry::capsSatisfy(gemmCapsMask(engine_), variant->requiredCapsMask))
		{
			OaLogDebug(oa::LogComponent::Engine,
				"Skipping %s (required GEMM caps unavailable on this device)", ent->name);
			++skipped;
			continue;
		}

		// register BOTH storage variants (DTYPE=0 FP32, DTYPE=1 BF16) so a dispatch can
		// select the one matching its operand tensors (oa::ComputeNode::dtype), instead of a
		// engine precision mode. DTYPE=1 is skipped when the device can't do native bf16. in
		// FP32 engine mode every tensor is FP32 → node dtype 0 → the =0 variant (unchanged);
		// only the extra =1 variants are loaded (unused). See oa::precisionDtype.md.
		const oa::U32 maxDtype = impl_->device_.nativeShaderBfloat16Usable() ? 1u : 0u;
		ShaderLoadPlan plan;
		plan.name = ent->name;
		plan.firstRequest = static_cast<oa::U32>(requests.size());
		plan.requestCount = maxDtype + 1u;
		for (oa::U32 dt = 0; dt <= maxDtype; ++dt) {
			oa::PipelineSpec spec{.numBindings = 16, .pushConstantBytes = 128,
				.specConstants = oa::Vec<oa::SpecConstant>{
					oa::SpecConstant{.id = 0, .value = dt}}};
			requests.pushBack(oa::PipelineLoadRequest{
				.name = ent->name,
				.spirv = oa::Span<const oa::U8>(ent->data, ent->size),
				.spec = oa::move(spec),
			});
		}
		plans.pushBack(plan);
	}

	const oa::I64 configuredThreads = oa::EnvFlag::getInt("OA_SHADER_LOAD_THREADS", 0);
	const oa::U32 hardwareThreads = oa::Thread::hardwareConcurrency();
	// A populated vulkan pipeline cache makes pipeline creation extremely cheap;
	// cloning and merging it per worker costs more than serial creation. Cold
	// driver compilation is CPU-heavy and scales best around physical-core count,
	// approximated portably as half of logical CPUs and capped to avoid excessive
	// cache replication on large hosts. The environment override remains exact.
	oa::U32 loadThreads = 1;
	if (configuredThreads > 0) {
		loadThreads = static_cast<oa::U32>(oa::min<oa::I64>(configuredThreads, 64));
	} else if (!impl_->pipelines_.hasInitialCacheData()) {
		loadThreads = oa::min<oa::U32>(oa::max<oa::U32>(1u, hardwareThreads / 2u), 8u);
	}
	loadThreads = oa::max<oa::U32>(1u,
		oa::min<oa::U32>(loadThreads, static_cast<oa::U32>(requests.size())));

	OaLogInfo(oa::LogComponent::Engine,
		"Preloading %zu shader pipelines (%u thread%s, %s cache)",
		requests.size(), loadThreads, loadThreads == 1 ? "" : "s",
		impl_->pipelines_.hasInitialCacheData() ? "warm" : "cold");

	const auto loadBegin = oa::steadyNow();
	oa::Vec<oa::Status> requestStatuses;
	if (!requests.empty()) {
		(void)impl_->pipelines_.ensurePipelinesParallel(
			impl_->device_,
			oa::Span<const oa::PipelineLoadRequest>(requests.data(), requests.size()),
			loadThreads,
			&requestStatuses);
	}
	const oa::F64 loadMs = (oa::steadyNow() - loadBegin).toMilliseconds();

	for (const auto& plan : plans) {
		oa::Status shaderStatus = oa::Status::ok();
		for (oa::U32 offset = 0; offset < plan.requestCount; ++offset) {
			const oa::Status& status = requestStatuses[plan.firstRequest + offset];
			if (status.isError()) shaderStatus = status;
		}
		if (shaderStatus.isOk()) {
			++loaded;
		} else {
			++failed;
			OaLogWarn(oa::LogComponent::Engine, "Failed to load shader '%s': %s",
				plan.name, shaderStatus.getMessage().cStr());
		}
	}

	OaLogInfo(oa::LogComponent::Engine,
		"Loaded %u/%u shaders (skipped=%u, failed=%u, pipelines=%zu, threads=%u, %.2f ms, precision=%s)",
		loaded, total, skipped, failed, requests.size(), loadThreads, loadMs,
		oa::precisionToString(engine_.getPrecision()));

	if (loaded == 0) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"build-generated SPIR-V registry contains no loadable compute pipelines");
	}
	return oa::Status::ok();
}

// ─── Buffer allocation ─────────────────────────────────────────────────────

oa::Result<oavk::Buffer> oa::EngineAccess::allocBuffer(oa::U64 inSize) {
	return allocBuffer(inSize, oa::MemoryPlacement::HostUpload);
}

oa::Result<oavk::Buffer> oa::EngineAccess::allocBuffer(
	oa::U64 inSize, oa::MemoryPlacement inPlacement)
{
	const oa::MemoryPlacement placement = inPlacement == oa::MemoryPlacement::Auto
		? impl_->matrixPlacement_
		: inPlacement;
	if (placement == oa::MemoryPlacement::DeviceLocal) {
		return allocBufferDevice(inSize);
	}
	if (placement == oa::MemoryPlacement::Unified) {
		return allocBufferBar(inSize);
	}
	if (placement == oa::MemoryPlacement::HostReadback) {
		auto result = impl_->allocator_.allocHostReadback(inSize);
		if (result) {
			const auto admission =
				oavk::validateStorageBufferDescriptor(impl_->device_, *result);
			if (not admission.isOk()) {
				impl_->allocator_.free(*result);
				return admission;
			}
			if (registerBuffer(*result) == OA_BINDLESS_INVALID) {
				impl_->allocator_.free(*result);
				return oa::Status::error(oa::StatusCode::ResourceExhausted,
					"allocBuffer: bindless buffer heap exhausted");
			}
		}
		return result;
	}

	{
		oa::ScopedLock lock(impl_->hostVisibleBufferCacheMutex_);
		oa::Usize bestIndex = impl_->hostVisibleBufferCache_.size();
		oa::U64 bestCapacity = static_cast<oa::U64>(-1);
		for (oa::Usize index = 0; index < impl_->hostVisibleBufferCache_.size(); ++index) {
			const oa::U64 capacity = impl_->hostVisibleBufferCache_[index].capacity;
			if (capacity >= inSize && capacity < bestCapacity) {
				bestIndex = index;
				bestCapacity = capacity;
			}
		}
		if (bestIndex != impl_->hostVisibleBufferCache_.size()) {
			oavk::Buffer reused = oa::move(
				impl_->hostVisibleBufferCache_[bestIndex].buffer);
			impl_->hostVisibleBufferCache_.erase(
				impl_->hostVisibleBufferCache_.begin() + bestIndex);
			impl_->hostVisibleBufferCacheBytes_ -= reused.capacity;
			reused.size = inSize;
			const auto update = updateBufferDescriptor(reused);
			if (not update.isOk()) {
				deregisterBuffer(reused);
				impl_->allocator_.free(reused);
				return update;
			}
			return reused;
		}
	}

	auto res = impl_->allocator_.allocHostVisible(inSize);
	if (!res) {
		return res;
	}
	const auto admission =
		oavk::validateStorageBufferDescriptor(impl_->device_, *res);
	if (not admission.isOk()) {
		impl_->allocator_.free(*res);
		return admission;
	}
	if (registerBuffer(*res) == OA_BINDLESS_INVALID) {
		impl_->allocator_.free(*res);
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"allocBuffer: bindless buffer heap exhausted");
	}
	return res;
}

oa::Status oa::EngineAccess::uploadBuffer(
	const oavk::Buffer& inDst, oa::U64 inDstOffset, const void* inData, oa::U64 inSize)
{
	if (!inDst.buffer || !inData || inSize == 0 || inDstOffset > inDst.size
		|| inSize > inDst.size - inDstOffset) {
		return oa::Status::invalidArgument("uploadBuffer: invalid source or destination range");
	}
	if (inDst.mappedPtr) {
		auto* destination =
			static_cast<oa::U8*>(inDst.mappedPtr) + inDstOffset;
		// A caller that filled the mapped range in place may use uploadBuffer as
		// the semantic host-write publication boundary. Avoid invoking memcpy on
		// identical source and destination ranges.
		if (destination != inData) {
			oa::memcpy(destination, inData, static_cast<oa::Usize>(inSize));
		}
		if (not impl_->allocator_.flushHostBuffer(inDst, inDstOffset, inSize)) {
			return oa::Status::error(
				oa::StatusCode::VulkanError, "uploadBuffer: mapped flush failed");
		}
		inDst.markMutation();
		return oa::Status::ok();
	}

	// vkCmdCopyBuffer requires four-byte aligned offsets and sizes. matrices may
	// still contain byte/half scalars or expose an unaligned view, so promote the
	// transfer to the enclosing words. Partial updates preserve neighbouring
	// bytes through a read-modify-write; whole-buffer uploads simply zero pad the
	// physical tail allocated by OaVma.
	oavk::Buffer copyDst = inDst;
	oa::U64 copyOffset = inDstOffset;
	const void* copyData = inData;
	oa::U64 copySize = inSize;
	oa::Vec<oa::U8> alignedData;
	if ((copyOffset & 3ULL) != 0 || (copySize & 3ULL) != 0) {
		const oa::U64 alignedBegin = copyOffset & ~3ULL;
		const oa::U64 alignedEnd = (copyOffset + copySize + 3ULL) & ~3ULL;
		if (alignedEnd > copyDst.capacity) {
			return oa::Status::invalidArgument("uploadBuffer: padded range exceeds capacity");
		}
		alignedData.resize(static_cast<oa::Usize>(alignedEnd - alignedBegin));
		if (copyOffset != 0 || copySize != inDst.size) {
			oavk::Buffer physical = inDst;
			physical.size = physical.capacity;
			OA_RETURN_IF_ERROR(readbackBuffer(physical, alignedBegin,
				alignedData.data(), alignedEnd - alignedBegin));
		} else {
			oa::memzero(alignedData.data(), alignedData.size());
		}
		oa::memcpy(alignedData.data() + (copyOffset - alignedBegin), copyData,
			static_cast<oa::Usize>(copySize));
		copyDst.size = copyDst.capacity;
		copyOffset = alignedBegin;
		copyData = alignedData.data();
		copySize = alignedEnd - alignedBegin;
	}

	oa::ScopedLock lock(impl_->uploadRingMutex_);
	if (!impl_->uploadRing_ || impl_->uploadRing_->frameCapacityBytes() < copySize) {
		if (impl_->uploadRing_) {
			OA_RETURN_IF_ERROR(impl_->uploadRing_->close());
			impl_->uploadRing_.reset();
		}
		const oa::U64 frameBytes = (copySize + 255ULL) & ~255ULL;
		const oa::U64 capacity = oa::max<oa::U64>(64ULL * 1024ULL * 1024ULL, frameBytes * 3ULL);
		auto ring = oa::UploadRing::create(engine_, oa::UploadRingConfig{
			.capacityBytes = capacity,
			.framesInFlight = 3,
			.alignment = 256,
		});
		if (!ring) return ring.getStatus();
		impl_->uploadRing_ = oa::makeUnique<oa::UploadRing>(oa::move(*ring));
	}
	OA_RETURN_IF_ERROR(impl_->uploadRing_->beginBatch());
	OA_RETURN_IF_ERROR(impl_->uploadRing_->upload(
		copyDst, copyOffset, copyData, copySize));
	auto completion = impl_->uploadRing_->submit();
	if (!completion) return completion.getStatus();
	const auto status = completion->wait();
	if (status.isOk()) inDst.markMutation();
	return status;
}

oa::Status oa::EngineAccess::readbackBuffer(
	const oavk::Buffer& inSrc, oa::U64 inSrcOffset, void* outData, oa::U64 inSize)
{
	if (not inSrc.buffer or not outData or inSize == 0 or inSrcOffset > inSrc.size
		or inSize > inSrc.size - inSrcOffset) {
		return oa::Status::invalidArgument("readbackBuffer: invalid source or destination range");
	}
	if (inSrc.allocation == nullptr or inSrc.aliasIdentity != nullptr
		or inSrc.allocatorIdentity != impl_->allocator_.allocator) {
		return oa::Status::invalidArgument(
			"readbackBuffer: source must be a non-aliased engine VMA allocation");
	}

	oa::ScopedLock lock(impl_->readbackMutex_);
	if (not impl_->readbackStream_.commandPool) {
		auto streamResult = oavk::Stream::create(
			impl_->device_, impl_->device_.queues.computeQueueFamily, impl_->device_.queues.computeQueue);
		if (not streamResult) return streamResult.getStatus();
		impl_->readbackStream_ = oa::move(*streamResult);
	}
	if (inSrc.mappedPtr) {
		oa::Status status = impl_->readbackStream_.begin(impl_->device_);
		if (status.isOk()) {
			impl_->readbackStream_.recordHostReadbackBarrier();
			status = impl_->readbackStream_.submit(engine_);
		}
		if (status.isOk()) status = impl_->readbackStream_.synchronize(impl_->device_);
		if (not status.isOk()) return status;
		if (not impl_->allocator_.invalidateHostBuffer(inSrc, inSrcOffset, inSize)) {
			return oa::Status::error(oa::StatusCode::VulkanError, "readbackBuffer: mapped invalidate failed");
		}
		oa::memcpy(outData, static_cast<const oa::U8*>(inSrc.mappedPtr) + inSrcOffset,
			static_cast<oa::Usize>(inSize));
		return oa::Status::ok();
	}
	const oa::U64 copyOffset = inSrcOffset & ~3ULL;
	const oa::U64 copyEnd = (inSrcOffset + inSize + 3ULL) & ~3ULL;
	if (copyEnd > inSrc.capacity) {
		return oa::Status::invalidArgument("readbackBuffer: padded range exceeds capacity");
	}
	const oa::U64 copySize = copyEnd - copyOffset;

	if (not impl_->readbackStaging_.buffer or impl_->readbackStaging_.capacity < copySize) {
		// begin() below would also wait before command-buffer reuse, but an old
		// staging allocation cannot be released until its previous copy completes.
		if (impl_->readbackStream_.submitted) {
			OA_RETURN_IF_ERROR(impl_->readbackStream_.synchronize(impl_->device_));
		}
		impl_->allocator_.free(impl_->readbackStaging_);
		oa::U64 capacity = 64ULL * 1024ULL;
		while (capacity < copySize and capacity <= UINT64_MAX / 2ULL) capacity *= 2ULL;
		if (capacity < copySize) capacity = copySize;
		auto readbackResult = impl_->allocator_.allocHostReadback(capacity);
		if (not readbackResult) return readbackResult.getStatus();
		impl_->readbackStaging_ = oa::move(*readbackResult);
	}

	oa::Status status = impl_->readbackStream_.begin(impl_->device_);
	if (status.isOk()) {
		const oavk::BufferCopyRegion region{
			.srcOffset = copyOffset,
			.dstOffset = 0,
			.size = copySize,
		};
		impl_->readbackStream_.recordTransferReadBarrier(
			inSrc, copyOffset, copySize);
		impl_->readbackStream_.recordCopyBufferRegions(inSrc, impl_->readbackStaging_,
			oa::Span<const oavk::BufferCopyRegion>(&region, 1));
		impl_->readbackStream_.recordTransferWriteBarrier(
			impl_->readbackStaging_, 0U, copySize);
		status = impl_->readbackStream_.submit(engine_);
	}
	if (status.isOk()) status = impl_->readbackStream_.synchronize(impl_->device_);
	if (status.isOk() and not impl_->allocator_.invalidateHostBuffer(impl_->readbackStaging_, 0, copySize)) {
		status = oa::Status::error(oa::StatusCode::VulkanError, "readbackBuffer: staging invalidate failed");
	}
	if (status.isOk()) {
		oa::memcpy(outData,
			static_cast<const oa::U8*>(impl_->readbackStaging_.mappedPtr) + (inSrcOffset - copyOffset),
			static_cast<oa::Usize>(inSize));
	}
	return status;
}

oa::Result<oavk::Buffer> oa::EngineAccess::allocBufferDevice(oa::U64 inSize) {
	auto res = impl_->allocator_.allocDevice(inSize);
	if (!res) {
		return res;
	}
	const auto admission =
		oavk::validateStorageBufferDescriptor(impl_->device_, *res);
	if (not admission.isOk()) {
		impl_->allocator_.free(*res);
		return admission;
	}
	if (registerBuffer(*res) == OA_BINDLESS_INVALID) {
		impl_->allocator_.free(*res);
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"allocBufferDevice: bindless buffer heap exhausted");
	}
	return res;
}

oa::Result<oavk::Buffer> oa::EngineAccess::allocBufferBar(oa::U64 inSize) {
	auto res = impl_->allocator_.allocBar(inSize);
	if (!res) {
		return res;
	}
	const auto admission =
		oavk::validateStorageBufferDescriptor(impl_->device_, *res);
	if (not admission.isOk()) {
		impl_->allocator_.free(*res);
		return admission;
	}
	if (registerBuffer(*res) == OA_BINDLESS_INVALID) {
		impl_->allocator_.free(*res);
		return oa::Status::error(oa::StatusCode::ResourceExhausted,
			"allocBufferBar: bindless buffer heap exhausted");
	}
	return res;
}

void oa::EngineAccess::freeBuffer(oavk::Buffer& inOutBuffer) {
	if (inOutBuffer.buffer && inOutBuffer.mappedPtr
		&& inOutBuffer.placement == oa::MemoryPlacement::HostUpload
		&& !inOutBuffer.isBar()
		&& !inOutBuffer.isTransient()
		&& (inOutBuffer.flags & OA_VK_BUFFER_FLAG_ALIAS) == 0 &&
		inOutBuffer.bindlessIndex != OA_BINDLESS_INVALID) {
		oa::ScopedLock lock(impl_->hostVisibleBufferCacheMutex_);
		const oa::Bool canCache =
			impl_->hostVisibleBufferCache_.size() < kHostVisibleCacheMaxBuffers &&
			inOutBuffer.capacity <= kHostVisibleCacheMaxBytes &&
			impl_->hostVisibleBufferCacheBytes_ + inOutBuffer.capacity <= kHostVisibleCacheMaxBytes;
		if (canCache) {
			impl_->hostVisibleBufferCacheBytes_ += inOutBuffer.capacity;
			impl_->hostVisibleBufferCache_.pushBack({
				inOutBuffer.capacity,
				inOutBuffer
			});
			inOutBuffer = oavk::Buffer{};
			return;
		}
	}
	deregisterBuffer(inOutBuffer);
	impl_->allocator_.free(inOutBuffer);
}

// ─── bindless ──────────────────────────────────────────────────────────────

oa::U32 oa::EngineAccess::registerBuffer(oavk::Buffer& inOutBuffer) {
	if (!impl_->bindless_.descriptorSet) {
		return OA_BINDLESS_INVALID;
	}
	oa::U32 idx = impl_->bindless_.registerBuffer(impl_->device_, inOutBuffer);
	inOutBuffer.bindlessIndex = idx;
	return idx;
}

oa::Status oa::EngineAccess::updateBufferDescriptor(const oavk::Buffer& inBuffer) {
	if (inBuffer.bindlessIndex == OA_BINDLESS_INVALID) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"updateBufferDescriptor: buffer is not registered");
	}
	return impl_->bindless_.update(impl_->device_, inBuffer.bindlessIndex, inBuffer);
}

void oa::EngineAccess::deregisterBuffer(oavk::Buffer& inOutBuffer) {
	if (inOutBuffer.bindlessIndex == OA_BINDLESS_INVALID) {
		return;
	}
	impl_->bindless_.deregister(inOutBuffer.bindlessIndex);
	inOutBuffer.bindlessIndex = OA_BINDLESS_INVALID;
}

// ─── stream pool ───────────────────────────────────────────────────────────

oavk::Stream* oa::EngineAccess::acquireStream() {
	collectRetiredImageDispatches();
	collectRetiredExecutionPlans();
	collectRetiredSessionBatches();
	oa::SpinlockGuard guard(impl_->streamPoolLock_);
	if (!impl_->freeStack_.empty()) {
		oa::U32 idx = impl_->freeStack_.back();
		impl_->freeStack_.popBack();
		return impl_->streamPool_[idx].get();
	}
	auto res = oavk::Stream::createCompute(impl_->device_);
	if (!res) {
		return nullptr;
	}
	auto ptr = oa::makeUnique<oavk::Stream>(oa::move(*res));
	oavk::Stream* raw = ptr.get();
	impl_->streamPool_.pushBack(oa::move(ptr));
	return raw;
}

bool oa::Engine::ownsEvent(const oa::Event& inEvent) const noexcept {
	return inEvent.isValid()
		and oa::EventAccess::deviceIdentity(inEvent) == &impl_->device_;
}

void oa::EngineAccess::collectRetiredGraphicsStreams() {
	if (impl_->device_.device == nullptr) return;
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	for (auto& slot : impl_->graphicsStreamPool_) {
		if (slot.state != Impl::GraphicsStreamSlotState::Retired
			or not slot.stream
			or not slot.stream->isComplete(impl_->device_)) {
			continue;
		}
		slot.stream->submitted = false;
		slot.completion = {};
		slot.state = Impl::GraphicsStreamSlotState::Free;
	}
}

oavk::Stream* oa::EngineAccess::graphicsStreamForLease(
	oa::U32 inSlot, oa::U64 inGeneration) noexcept
{
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	if (inSlot >= impl_->graphicsStreamPool_.size()) return nullptr;
	auto& slot = impl_->graphicsStreamPool_[inSlot];
	if (slot.generation != inGeneration
		or slot.state != Impl::GraphicsStreamSlotState::Recording) {
		return nullptr;
	}
	return slot.stream.get();
}

oa::Result<oa::Event> oa::EngineAccess::submitGraphicsStream(
	oa::U32 inSlot,
	oa::U64 inGeneration,
	oa::Span<const oa::Event> inDependencies)
{
	if (not engine_.isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics submission requires a ready engine");
	}
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	if (inSlot >= impl_->graphicsStreamPool_.size()) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease slot is invalid");
	}
	auto& slot = impl_->graphicsStreamPool_[inSlot];
	if (slot.generation != inGeneration
		or slot.state != Impl::GraphicsStreamSlotState::Recording
		or not slot.stream) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease is stale or not recording");
	}

	oa::Vec<oavk::TimelineWait> waits;
	waits.reserve(inDependencies.size());
	for (const oa::Event& dependency : inDependencies) {
		if (not engine_.ownsEvent(dependency)) {
			return oa::Status::error(
				oa::StatusCode::InvalidArgument,
				"graphics dependency event belongs to another engine");
		}
		if (not dependency.hasQueueFamily()) {
			return oa::Status::error(
				oa::StatusCode::InvalidArgument,
				"graphics dependency event has no producer queue-family provenance");
		}
		if (dependency.queueFamily()
			!= impl_->device_.queues.graphicsQueueFamily) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"cross-family graphics dependencies require explicit resource ownership transfer");
		}
		waits.pushBack(oa::EventAccess::timelineWait(dependency));
	}

	const oa::Status submitStatus = waits.empty()
		? slot.stream->submit(engine_)
		: slot.stream->submitWithDependencies(
			engine_, oa::Span<const oavk::TimelineWait>(waits.data(), waits.size()));
	if (not submitStatus.isOk()) {
		const oa::Status resetStatus = slot.stream->resetUnsubmitted(impl_->device_);
		slot.completion = {};
		slot.state = resetStatus.isOk()
			? Impl::GraphicsStreamSlotState::Free
			: Impl::GraphicsStreamSlotState::Quarantined;
		return resetStatus.isOk() ? submitStatus : resetStatus;
	}

	const oa::Event completion = slot.stream->completion(impl_->device_);
	if (not completion.isValid()) {
		slot.state = Impl::GraphicsStreamSlotState::Retired;
		return oa::Status::error(
			oa::StatusCode::Internal,
			"graphics submission did not produce an exact completion event");
	}
	slot.completion = completion;
	slot.state = Impl::GraphicsStreamSlotState::Submitted;
	return completion;
}

oa::Status oa::EngineAccess::cancelGraphicsStream(
	oa::U32 inSlot, oa::U64 inGeneration)
{
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	if (inSlot >= impl_->graphicsStreamPool_.size()) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease slot is invalid");
	}
	auto& slot = impl_->graphicsStreamPool_[inSlot];
	if (slot.generation != inGeneration
		or slot.state != Impl::GraphicsStreamSlotState::Recording
		or not slot.stream) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"only an exact active graphics recording can be cancelled");
	}
	const oa::Status status = slot.stream->resetUnsubmitted(impl_->device_);
	slot.completion = {};
	slot.state = status.isOk()
		? Impl::GraphicsStreamSlotState::Free
		: Impl::GraphicsStreamSlotState::Quarantined;
	return status;
}

oa::Status oa::EngineAccess::recycleGraphicsStream(
	oa::U32 inSlot,
	oa::U64 inGeneration,
	const oa::Event& inCompletion)
{
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	if (inSlot >= impl_->graphicsStreamPool_.size()) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease slot is invalid");
	}
	auto& slot = impl_->graphicsStreamPool_[inSlot];
	if (slot.generation != inGeneration
		or slot.state != Impl::GraphicsStreamSlotState::Submitted
		or not slot.stream) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease is stale or not submitted");
	}
	if (not engine_.ownsEvent(inCompletion)
		or not slot.completion.isSameCompletion(inCompletion)) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream recycle requires its exact completion event");
	}
	if (not inCompletion.isComplete()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"graphics stream completion is still pending");
	}
	slot.stream->submitted = false;
	slot.completion = {};
	slot.state = Impl::GraphicsStreamSlotState::Free;
	return oa::Status::ok();
}

oa::Status oa::EngineAccess::abandonGraphicsStream(
	oa::U32 inSlot, oa::U64 inGeneration)
{
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	if (inSlot >= impl_->graphicsStreamPool_.size()) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease slot is invalid");
	}
	auto& slot = impl_->graphicsStreamPool_[inSlot];
	if (slot.generation != inGeneration or not slot.stream) {
		return oa::Status::error(
			oa::StatusCode::InvalidArgument,
			"graphics stream lease generation is stale");
	}
	if (slot.state == Impl::GraphicsStreamSlotState::Recording) {
		const oa::Status status = slot.stream->resetUnsubmitted(impl_->device_);
		slot.completion = {};
		slot.state = status.isOk()
			? Impl::GraphicsStreamSlotState::Free
			: Impl::GraphicsStreamSlotState::Quarantined;
		return status;
	}
	if (slot.state == Impl::GraphicsStreamSlotState::Submitted) {
		if (slot.stream->isComplete(impl_->device_)) {
			slot.stream->submitted = false;
			slot.completion = {};
			slot.state = Impl::GraphicsStreamSlotState::Free;
		} else {
			slot.state = Impl::GraphicsStreamSlotState::Retired;
		}
		return oa::Status::ok();
	}
	if (slot.state == Impl::GraphicsStreamSlotState::Retired
		or slot.state == Impl::GraphicsStreamSlotState::Quarantined
		or slot.state == Impl::GraphicsStreamSlotState::Free) {
		return oa::Status::ok();
	}
	return oa::Status::error(
		oa::StatusCode::InvalidArgument,
		"graphics stream lease is no longer active");
}

oa::Status oa::EngineAccess::completeGraphicsStreams() {
	oa::ScopedLock lock(impl_->graphicsStreamPoolMutex_);
	oa::Status firstError = oa::Status::ok();
	for (auto& slot : impl_->graphicsStreamPool_) {
		if (not slot.stream) continue;
		if (slot.state == Impl::GraphicsStreamSlotState::Recording) {
			const oa::Status status = slot.stream->resetUnsubmitted(impl_->device_);
			if (firstError.isOk() and not status.isOk()) firstError = status;
			slot.state = status.isOk()
				? Impl::GraphicsStreamSlotState::Free
				: Impl::GraphicsStreamSlotState::Quarantined;
			continue;
		}
		if (slot.state != Impl::GraphicsStreamSlotState::Submitted
			and slot.state != Impl::GraphicsStreamSlotState::Retired) {
			continue;
		}
		const oa::Status status = slot.stream->synchronize(impl_->device_);
		if (not status.isOk()) {
			if (firstError.isOk()) firstError = status;
			// synchronize leaves Submitted set when its wait fails. Preserve the
			// exact Submitted/Retired state and completion event so Close can retry
			// instead of quarantining live work and then tearing its resources down.
			continue;
		}
		slot.completion = {};
		slot.state = Impl::GraphicsStreamSlotState::Free;
	}
	return firstError;
}

void oa::EngineAccess::retireImageDispatch(oa::RetiredImageDispatch&& inRetired)
{
	if (inRetired.stream == nullptr) return;
	oa::ScopedLock lock(impl_->retiredImageDispatchMutex_);
	impl_->retiredImageDispatches_.pushBack(oa::move(inRetired));
}

void oa::EngineAccess::collectRetiredImageDispatches()
{
	oa::ScopedLock lock(impl_->retiredImageDispatchMutex_);
	for (oa::Usize i = impl_->retiredImageDispatches_.size(); i > 0; --i) {
		auto& retired = impl_->retiredImageDispatches_[i - 1];
		if (retired.stream == nullptr || !retired.stream->isComplete(impl_->device_)) {
			continue;
		}
		for (VkImageView view : retired.imageViews) {
			if (view != VK_NULL_HANDLE) {
				impl_->device_.deviceDispatch.vkDestroyImageView(
					static_cast<VkDevice>(impl_->device_.device), view, nullptr);
			}
		}
		for (oa::U32 idx : retired.storageImageSlots) impl_->bindless_.deregisterStorageImage(idx);
		for (oa::U32 idx : retired.sampledImageSlots) impl_->bindless_.deregisterSampledImage(idx);
		for (oa::U32 idx : retired.samplerSlots) impl_->bindless_.deregisterSampler(idx);
		retired.stream->submitted = false;
		releaseStream(retired.stream);
		impl_->retiredImageDispatches_.erase(impl_->retiredImageDispatches_.begin() + (i - 1));
	}
}

void oa::EngineAccess::retireExecutionPlan(oa::UniquePtr<oa::ExecutableGraph>&& inGraph)
{
	if (not inGraph) return;
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredExecutionPlanMutex_);
	impl_->retiredExecutionPlans_.pushBack(oa::move(inGraph));
}

void oa::EngineAccess::collectRetiredExecutionPlans()
{
	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredExecutionPlanMutex_);
		retired = oa::move(impl_->retiredExecutionPlans_);
	}

	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> pending;
	for (auto& graph : retired) {
		const auto completion = graph->lastCompletion(engine_);
		if (completion.isValid() and not completion.isComplete()) {
			pending.pushBack(oa::move(graph));
			continue;
		}
	}
	if (not pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredExecutionPlanMutex_);
		for (auto& graph : pending) {
			impl_->retiredExecutionPlans_.pushBack(oa::move(graph));
		}
	}
}

oa::Status oa::EngineAccess::completeRetiredExecutionPlans()
{
	oa::Status result = oa::Status::ok();
	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredExecutionPlanMutex_);
		retired = oa::move(impl_->retiredExecutionPlans_);
	}

	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>> pending;
	for (auto& graph : retired) {
		const auto waitStatus = graph->waitForPendingReplay(engine_);
		if (not waitStatus.isOk()) {
			if (result.isOk()) result = waitStatus;
			pending.pushBack(oa::move(graph));
			continue;
		}
	}
	if (not pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredExecutionPlanMutex_);
		for (auto& graph : pending) {
			impl_->retiredExecutionPlans_.pushBack(oa::move(graph));
		}
	}
	return result;
}

void oa::EngineAccess::retireSessionBatch(
	oavk::Stream* inStream,
	const oa::Event& inCompletion,
	oa::Vec<oa::UniquePtr<oa::ExecutableGraph>>&& inGraphs)
{
	if (inStream == nullptr) return;
	Impl::RetiredSessionBatch retired;
	retired.stream = inStream;
	retired.completion = inCompletion;
	retired.graphs = oa::move(inGraphs);
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredSessionBatchMutex_);
	impl_->retiredSessionBatches_.pushBack(oa::move(retired));
}

void oa::EngineAccess::collectRetiredSessionBatches()
{
	oa::Vec<Impl::RetiredSessionBatch> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredSessionBatchMutex_);
		retired = oa::move(impl_->retiredSessionBatches_);
	}

	oa::Vec<Impl::RetiredSessionBatch> pending;
	for (auto& batch : retired) {
		// The stream owns the timeline and remains the authoritative retirement
		// proof even if a malformed completion event reached this fallback path.
		if (batch.stream != nullptr and not batch.stream->isComplete(impl_->device_)) {
			pending.pushBack(oa::move(batch));
			continue;
		}
		if (batch.stream != nullptr) {
			batch.stream->submitted = false;
			const auto resetStatus =
				batch.stream->resetUnsubmitted(impl_->device_);
			if (not resetStatus.isOk()) {
				OaLogError(oa::LogComponent::Engine,
					"retired context stream reset failed: %s",
					resetStatus.getMessage().cStr());
				pending.pushBack(oa::move(batch));
				continue;
			}
			releaseStream(batch.stream);
		}
	}

	if (not pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredSessionBatchMutex_);
		for (auto& batch : pending) {
			impl_->retiredSessionBatches_.pushBack(oa::move(batch));
		}
	}
}

oa::Status oa::EngineAccess::completeRetiredSessionBatches()
{
	oa::Status result = oa::Status::ok();
	oa::Vec<Impl::RetiredSessionBatch> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredSessionBatchMutex_);
		retired = oa::move(impl_->retiredSessionBatches_);
	}

	oa::Vec<Impl::RetiredSessionBatch> pending;
	for (auto& batch : retired) {
		const auto waitStatus = batch.stream != nullptr
			? batch.stream->synchronize(impl_->device_)
			: oa::Status::error(oa::StatusCode::Internal,
				"retired context batch has no stream");
		if (not waitStatus.isOk()) {
			if (result.isOk()) result = waitStatus;
			pending.pushBack(oa::move(batch));
			continue;
		}
		if (batch.stream != nullptr) {
			const auto resetStatus =
				batch.stream->resetUnsubmitted(impl_->device_);
			if (not resetStatus.isOk()) {
				if (result.isOk()) result = resetStatus;
				pending.pushBack(oa::move(batch));
				continue;
			}
			releaseStream(batch.stream);
		}
	}

	if (not pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredSessionBatchMutex_);
		for (auto& batch : pending) {
			impl_->retiredSessionBatches_.pushBack(oa::move(batch));
		}
	}
	return result;
}

void oa::EngineAccess::retireUploadRing(oa::UniquePtr<oa::RetiredUploadRing>&& inRing)
{
	if (not inRing) return;
	oa::ScopedLock<oa::Mutex> lock(impl_->retiredUploadRingMutex_);
	impl_->retiredUploadRings_.pushBack(oa::move(inRing));
}

oa::Status oa::EngineAccess::completeRetiredUploadRings()
{
	oa::Status result = oa::Status::ok();
	oa::Vec<oa::UniquePtr<oa::RetiredUploadRing>> retired;
	{
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredUploadRingMutex_);
		retired = oa::move(impl_->retiredUploadRings_);
	}

	oa::Vec<oa::UniquePtr<oa::RetiredUploadRing>> pending;
	for (auto& ring : retired) {
		oa::Bool complete = true;
		for (auto& frame : ring->frames) {
			if (not frame) continue;
			const auto waitStatus = frame->stream.synchronize(impl_->device_);
			if (not waitStatus.isOk()) {
				if (result.isOk()) result = waitStatus;
				complete = false;
			}
		}
		if (not complete) {
			pending.pushBack(oa::move(ring));
			continue;
		}
		for (auto& frame : ring->frames) {
			if (frame) frame->stream.destroy(impl_->device_);
		}
		freeBuffer(ring->staging);
	}
	if (not pending.empty()) {
		oa::ScopedLock<oa::Mutex> lock(impl_->retiredUploadRingMutex_);
		for (auto& ring : pending) {
			impl_->retiredUploadRings_.pushBack(oa::move(ring));
		}
	}
	return result;
}

void oa::EngineAccess::releaseStream(oavk::Stream* inStream) {
	oa::SpinlockGuard guard(impl_->streamPoolLock_);
	for (oa::U32 i = 0; i < impl_->streamPool_.size(); ++i) {
		if (impl_->streamPool_[i].get() == inStream) {
			impl_->freeStack_.pushBack(i);
			return;
		}
	}
}

// ─── Async Compute stream pool ─────────────────────────────────────────────

oavk::Stream* oa::EngineAccess::acquireAsyncStream() {
	if (!impl_->device_.queues.hasAsyncCompute) {
		return acquireStream();
	}
	oa::SpinlockGuard guard(impl_->asyncStreamPoolLock_);
	if (!impl_->asyncFreeStack_.empty()) {
		oa::U32 idx = impl_->asyncFreeStack_.back();
		impl_->asyncFreeStack_.popBack();
		return impl_->asyncStreamPool_[idx].get();
	}
	auto res = oavk::Stream::create(
		impl_->device_, impl_->device_.queues.asyncComputeQueueFamily, impl_->device_.queues.asyncComputeQueue);
	if (!res) {
		return nullptr;
	}
	auto ptr = oa::makeUnique<oavk::Stream>(oa::move(*res));
	oavk::Stream* raw = ptr.get();
	impl_->asyncStreamPool_.pushBack(oa::move(ptr));
	return raw;
}

void oa::EngineAccess::releaseAsyncStream(oavk::Stream* inStream) {
	if (!impl_->device_.queues.hasAsyncCompute) { releaseStream(inStream); return; }
	oa::SpinlockGuard guard(impl_->asyncStreamPoolLock_);
	for (oa::U32 i = 0; i < impl_->asyncStreamPool_.size(); ++i) {
		if (impl_->asyncStreamPool_[i].get() == inStream) {
			impl_->asyncFreeStack_.pushBack(i);
			return;
		}
	}
}

// ─── Thread-Safe queue submit ──────────────────────────────────────────────

void* oa::EngineAccess::queueSubmitMutex(void* inQueue) noexcept {
	switch (oavk::classifyQueueSubmitRoute(impl_->device_.queues, inQueue)) {
	case oavk::QueueSubmitRoute::Compute:
		return &impl_->computeQueueMutex_;
	case oavk::QueueSubmitRoute::AsyncCompute:
		return &impl_->asyncComputeQueueMutex_;
	case oavk::QueueSubmitRoute::Transfer:
		return &impl_->transferQueueMutex_;
	case oavk::QueueSubmitRoute::Graphics:
		return &impl_->graphicsQueueMutex_;
	case oavk::QueueSubmitRoute::Present:
		return &impl_->presentQueueMutex_;
	case oavk::QueueSubmitRoute::Unknown:
		return nullptr;
	}
	return nullptr;
}

void oa::EngineAccess::lockQueueSubmit(void* inQueue) {
	if (void* mutex = queueSubmitMutex(inQueue)) {
		static_cast<oa::Mutex*>(mutex)->lock();
	}
}

void oa::EngineAccess::unlockQueueSubmit(void* inQueue) {
	if (void* mutex = queueSubmitMutex(inQueue)) {
		static_cast<oa::Mutex*>(mutex)->unlock();
	}
}

static const char* queueSubmitRouteName(oavk::QueueSubmitRoute inRoute) noexcept {
	switch (inRoute) {
	case oavk::QueueSubmitRoute::Compute: return "compute";
	case oavk::QueueSubmitRoute::AsyncCompute: return "async compute";
	case oavk::QueueSubmitRoute::Transfer: return "transfer";
	case oavk::QueueSubmitRoute::Graphics: return "graphics";
	case oavk::QueueSubmitRoute::Present: return "present";
	case oavk::QueueSubmitRoute::Unknown: return "unknown";
	}
	return "unknown";
}

oa::Status oa::EngineAccess::submitToQueue(void* inQueue, void* inSubmitInfo, void* inFence) {
	collectRetiredExecutionPlans();
	const oavk::QueueSubmitRoute route = oavk::classifyQueueSubmitRoute(impl_->device_.queues, inQueue);
	if (route == oavk::QueueSubmitRoute::Unknown
		or route == oavk::QueueSubmitRoute::Present
		or inSubmitInfo == nullptr) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "unknown queue");
	}
	void* mutex = queueSubmitMutex(inQueue);
	if (mutex == nullptr) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "unknown queue");
	}
	VkQueue queue = static_cast<VkQueue>(inQueue);
	VkFence fence = static_cast<VkFence>(inFence);
	const VkSubmitInfo* si = static_cast<const VkSubmitInfo*>(inSubmitInfo);
	oa::ScopedLock lock(*static_cast<oa::Mutex*>(mutex));
	const VkResult result = impl_->device_.deviceDispatch.vkQueueSubmit(
		queue, 1, si, fence);
	if (result != VK_SUCCESS) {
		const char* routeName = queueSubmitRouteName(route);
		OaLogError(oa::LogComponent::Engine,
			"vkQueueSubmit (%s) failed, VkResult=%d",
			routeName, static_cast<int>(result));
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			oa::String("vkQueueSubmit (") + routeName + ") failed");
	}
	return oa::Status::ok();
}

oa::Status oa::EngineAccess::submitToQueue2(void* inQueue, const void* inSubmitInfo) {
	collectRetiredExecutionPlans();
	const oavk::QueueSubmitRoute route = oavk::classifyQueueSubmitRoute(impl_->device_.queues, inQueue);
	if (route == oavk::QueueSubmitRoute::Unknown
		or route == oavk::QueueSubmitRoute::Present
		or inSubmitInfo == nullptr) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "unknown queue");
	}
	void* mutex = queueSubmitMutex(inQueue);
	if (mutex == nullptr) {
		return oa::Status::error(oa::StatusCode::InvalidArgument, "unknown queue");
	}
	VkQueue queue = static_cast<VkQueue>(inQueue);
	const auto* submitInfo = static_cast<const VkSubmitInfo2*>(inSubmitInfo);
	oa::ScopedLock lock(*static_cast<oa::Mutex*>(mutex));
	const VkResult result = impl_->device_.deviceDispatch.vkQueueSubmit2(
		queue, 1, submitInfo, VK_NULL_HANDLE);
	if (result != VK_SUCCESS) {
		const char* routeName = queueSubmitRouteName(route);
		OaLogError(oa::LogComponent::Engine,
			"vkQueueSubmit2 (%s) failed, VkResult=%d",
			routeName, static_cast<int>(result));
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			oa::String("vkQueueSubmit2 (") + routeName + ") failed");
	}
	return oa::Status::ok();
}

// ─── Async Transfer ────────────────────────────────────────────────────────

oa::Result<oa::Event> oa::EngineAccess::copyBufferAsync(
	const oavk::Buffer& inSrc,
	const oavk::Buffer& inDst,
	oa::U64 inSize
) {
	if (not engine_.isReady()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"copyBufferAsync: engine is not ready");
	}
	if (inSrc.buffer == nullptr or inDst.buffer == nullptr or inSize == 0U) {
		return oa::Status::invalidArgument(
			"copyBufferAsync: source, destination, and size must be valid");
	}
	if (inSrc.allocation == nullptr or inDst.allocation == nullptr
		or inSrc.allocatorIdentity != impl_->allocator_.allocator
		or inDst.allocatorIdentity != impl_->allocator_.allocator
		or inSrc.aliasIdentity != nullptr or inDst.aliasIdentity != nullptr) {
		return oa::Status::invalidArgument(
			"copyBufferAsync: buffers must be non-aliased engine VMA allocations");
	}
	if (inSize > inSrc.size or inSize > inDst.size
		or (inSrc.capacity != 0U and inSize > inSrc.capacity)
		or (inDst.capacity != 0U and inSize > inDst.capacity)) {
		return oa::Status::invalidArgument(
			"copyBufferAsync: copy size exceeds a logical or physical buffer range");
	}
	if (inSrc.synchronizationIdentity() == inDst.synchronizationIdentity()) {
		return oa::Status::invalidArgument(
			"copyBufferAsync: source and destination memory must not overlap");
	}

	oa::ScopedLock lock(impl_->transferStreamMutex_);
	if (not impl_->transferStream_.commandPool) {
		// OA buffers use exclusive queue-family ownership. Keep this generic
		// helper on the primary compute queue until graph-level release/acquire
		// transfers and access intents can make a separate transfer queue exact.
		auto res = oavk::Stream::create(
			impl_->device_,
			impl_->device_.queues.computeQueueFamily,
			impl_->device_.queues.computeQueue
		);
		if (not res) {
			return res.getStatus();
		}
		impl_->transferStream_ = oa::move(*res);
	}
	if (impl_->transferStream_.submitted) {
		if (not impl_->transferStream_.isComplete(impl_->device_)) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"copyBufferAsync: previous transfer is still in flight; wait its oa::Event");
		}
		impl_->transferStream_.submitted = false;
	}
	OA_RETURN_IF_ERROR(impl_->transferStream_.begin(impl_->device_));
	impl_->transferStream_.recordTransferReadBarrier(inSrc, 0U, inSize);
	impl_->transferStream_.recordCopyBuffer(inSrc, inDst, inSize);
	impl_->transferStream_.recordTransferWriteBarrier(inDst, 0U, inSize);
	OA_RETURN_IF_ERROR(impl_->transferStream_.submit(engine_));
	oa::Event completion = impl_->transferStream_.completion(impl_->device_);
	if (not completion.isValid()) {
		return oa::Status::error(oa::StatusCode::Internal,
			"copyBufferAsync: submission produced no completion event");
	}
	return completion;
}
