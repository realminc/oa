#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Init.h>
#include <Oa/Runtime/Topology.h>
#include <Oa/Runtime/UploadRing.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Runtime/MatmulTypes.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Validation.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Precision.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Device.h>

#include <cctype>
#include <chrono>
#include <thread>

// ─── Global Engine ─────────────────────────────────────────────────────────

static OaComputeEngine* GEngine = nullptr;

static constexpr OaU64 kHostVisibleCacheMaxBytes = 256ull * 1024ull * 1024ull;
static constexpr OaU32 kHostVisibleCacheMaxBuffers = 4096u;
static bool OaAsciiEqI(const char* InA, const char* InB);

static OaMemoryPlacement ResolveMatrixPlacement(const OaVkDevice& InDevice) {
	const OaString requested = OaEnvFlag::GetString("OA_MATRIX_MEMORY", "auto");
	if (OaAsciiEqI(requested.CStr(), "device") || OaAsciiEqI(requested.CStr(), "device-local")
		|| OaAsciiEqI(requested.CStr(), "vram")) {
		return OaMemoryPlacement::DeviceLocal;
	}
	if (OaAsciiEqI(requested.CStr(), "host") || OaAsciiEqI(requested.CStr(), "upload")
		|| OaAsciiEqI(requested.CStr(), "mapped")) {
		return OaMemoryPlacement::HostUpload;
	}
	if (OaAsciiEqI(requested.CStr(), "unified") || OaAsciiEqI(requested.CStr(), "bar")) {
		return OaMemoryPlacement::Unified;
	}
	return InDevice.Info.Hardware.DeviceType == OaDeviceType::VkDiscrete
		? OaMemoryPlacement::DeviceLocal
		: OaMemoryPlacement::HostUpload;
}

void OaComputeEngine::SetAsGlobal() {
	GEngine = this;
	OaFnMatrix::SetWeightDtype(OaPrecisionDtype(GetPrecision()));
	OaContext::SetDefault(Context_.get());
}

void OaComputeEngine::ClearGlobal() {
	if (GEngine == this) {
		if (OaContext::GetDefaultPtr() == Context_.get()) {
			OaContext::SetDefault(nullptr);
		}
		OaFnMatrix::SetWeightDtype(OaScalarType::Float32);
		GEngine = nullptr;
	}
}

OaComputeEngine* OaComputeEngine::GetGlobal() { return GEngine; }

OaContext& OaComputeEngine::GetContext() const noexcept {
	assert(Context_ && "OaComputeEngine::GetContext: engine is not initialized");
	return *Context_;
}

static OaU32 OaVkEnumIndexOrZero(const OaVkDevice& InDev) {
	return InDev.Info.Hardware.EnumerationIndex != OaVkEnumerationIndexUnset
		? InDev.Info.Hardware.EnumerationIndex
		: 0u;
}

enum class OaDeviceInitLogMode : OaU8 {
	Compact,
	Full,
	Off,
};

static bool OaAsciiEqI(const char* InA, const char* InB) {
	if (InA == nullptr || InB == nullptr) {
		return InA == InB;
	}
	while (*InA != '\0' && *InB != '\0') {
		const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(*InA)));
		const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(*InB)));
		if (a != b) {
			return false;
		}
		++InA;
		++InB;
	}
	return *InA == '\0' && *InB == '\0';
}

static OaDeviceInitLogMode OaGetDeviceInitLogMode() {
	const OaString env = OaEnvFlag::GetString("OA_LOG_DEVICE_INIT", "");
	if (!env.empty()) {
		const char* value = env.c_str();
		if (OaAsciiEqI(value, "full") ||
		    OaAsciiEqI(value, "debug") ||
		    OaAsciiEqI(value, "verbose") ||
		    OaAsciiEqI(value, "1") ||
		    OaAsciiEqI(value, "true") ||
		    OaAsciiEqI(value, "on")) {
			return OaDeviceInitLogMode::Full;
		}
		if (OaAsciiEqI(value, "compact") ||
		    OaAsciiEqI(value, "minimal") ||
		    OaAsciiEqI(value, "min")) {
			return OaDeviceInitLogMode::Compact;
		}
		if (OaAsciiEqI(value, "off") ||
		    OaAsciiEqI(value, "none") ||
		    OaAsciiEqI(value, "0") ||
		    OaAsciiEqI(value, "false") ||
		    OaAsciiEqI(value, "no")) {
			return OaDeviceInitLogMode::Off;
		}
	}
#ifdef NDEBUG
	return OaDeviceInitLogMode::Compact;
#else
	return OaDeviceInitLogMode::Full;
#endif
}

static void LogSelectedDevicesSummaryMesh(const OaDeviceMesh& InMesh) {
	OA_LOG_INFO(OaLogComponent::Core, "Selected devices: (");
	const OaU32 n = InMesh.NodeCount();
	OaU32 i = 0;
	while (i < n) {
		const OaString& nm = InMesh.Nodes[i].Device.Info.Hardware.DeviceName;
		OaU32 j = i + 1;
		while (j < n && InMesh.Nodes[j].Device.Info.Hardware.DeviceName == nm) {
			++j;
		}
		const OaU32 cnt = j - i;
		if (cnt == 1) {
			const auto& node = InMesh.Nodes[i];
			const auto& d = node.Device;
			const auto role = OaDeviceRoleName(node.Role);
			const auto dtn = OaDeviceTypeName(d.Info.Hardware.DeviceType);
			OA_LOG_INFO(OaLogComponent::Core,
				"  (%u) [Vk %u] (%.*s · %.*s): %s — %s",
				node.Index,
				static_cast<unsigned>(OaVkEnumIndexOrZero(d)),
				static_cast<int>(role.size()), role.data(),
				static_cast<int>(dtn.size()), dtn.data(),
				d.Info.Hardware.VendorName.c_str(),
				d.Info.Hardware.DeviceName.c_str()
			);
		} else {
			const auto& d0 = InMesh.Nodes[i].Device;
			const OaU32 vk0 = OaVkEnumIndexOrZero(d0);
			const OaU32 vkLast = OaVkEnumIndexOrZero(InMesh.Nodes[j - 1].Device);
			OA_LOG_INFO(OaLogComponent::Core,
				"  (%u-%u) [Vk %u-%u] (count: %u): %s — %s",
				static_cast<unsigned>(InMesh.Nodes[i].Index),
				static_cast<unsigned>(InMesh.Nodes[j - 1].Index),
				static_cast<unsigned>(vk0),
				static_cast<unsigned>(vkLast),
				static_cast<unsigned>(cnt),
				d0.Info.Hardware.VendorName.c_str(),
				nm.c_str());
		}
		i = j;
	}
	OA_LOG_INFO(OaLogComponent::Core, ")");
}

static void LogSelectedDevicesSummarySingle(const OaVkDevice& InDev) {
	OA_LOG_INFO(OaLogComponent::Core, "Selected devices: (");
	OA_LOG_INFO(OaLogComponent::Core,
		"  (0) [Vk %u]: %s — %s",
		static_cast<unsigned>(OaVkEnumIndexOrZero(InDev)),
		InDev.Info.Hardware.VendorName.c_str(),
		InDev.Info.Hardware.DeviceName.c_str());
	OA_LOG_INFO(OaLogComponent::Core, ")");
}

void OaComputeEngine::LogSelectedDevices() {
	const OaDeviceInitLogMode mode = OaGetDeviceInitLogMode();
	if (mode == OaDeviceInitLogMode::Off) {
		return;
	}
	if (mode == OaDeviceInitLogMode::Compact) {
		if (Mesh_) {
			for (OaU32 k = 0; k < Mesh_->NodeCount(); ++k) {
				Mesh_->Nodes[k].Device.PrintInfoCompact();
			}
			return;
		}
		Device.PrintInfoCompact();
		return;
	}

	if (Mesh_) {
		LogSelectedDevicesSummaryMesh(*Mesh_);
		for (OaU32 k = 0; k < Mesh_->NodeCount(); ++k) {
			auto& node = Mesh_->Nodes[k];
			const auto role = OaDeviceRoleName(node.Role);
			OA_LOG_INFO(OaLogComponent::Core,
				"OaVkDevice — mesh node %u · role %.*s",
				static_cast<unsigned>(node.Index),
				static_cast<int>(role.size()), role.data());
			node.Device.PrintInfoDetailed();
		}
		return;
	}
	LogSelectedDevicesSummarySingle(Device);
	Device.PrintInfoDetailed();
	if (OaEnvFlag::IsSet("OA_LOG_COOPMAT_SHAPES")) {
		OaVkLogCoopMatShapes(Device.Info.Software.CoopMatShapes, "      ");
	}
}

OaComputeEngine::~OaComputeEngine() { Destroy(); }

static OaU32 RegisterBufferOnMeshNode(OaDeviceNode& InNode, OaVkBuffer& InOutBuf) {
	InOutBuf.NodeIndex = InNode.Index;
	if (!InNode.Bindless.DescriptorSet) return OA_BINDLESS_INVALID;
	OaU32 idx = InNode.Bindless.Register(InNode.Device, InOutBuf);
	InOutBuf.BindlessIndex = idx;
	return idx;
}

// OaComputeEngine is pinned (move/copy = delete in the header). It owns a
// VkInstance/VkDevice/VMA/queues/mutexes and registers its own address as the
// process global, so it must never relocate. Create() builds it on the heap and
// returns an owning pointer; there is deliberately no move ctor/assignment here.
// (The old hand-written move surgery — copying the VMA handle, nulling the
// source's Device.Device to dodge a double-free, re-patching GEngine — is gone.)

OaResult<OaUniquePtr<OaComputeEngine>> OaComputeEngine::Create(const OaEngineConfig& InConfig) {
	// Pinned construction: build the engine on the heap so its address is stable
	// (it can register itself as the process-global engine during InitInPlace) and
	// return an owning pointer. No build-then-move.
	auto engine = OaMakeUniquePtr<OaComputeEngine>();
	OA_RETURN_IF_ERROR(engine->InitInPlace(InConfig));
	return engine;
}

OaStatus OaComputeEngine::InitInPlace(const OaEngineConfig& InConfig) {
	if (State_ != OaEngineState::Empty) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaComputeEngine::InitInPlace: engine initialization is one-shot");
	}
	State_ = OaEngineState::Initializing;
	auto status = InitInPlaceImpl(InConfig);
	State_ = status.IsOk() ? OaEngineState::Ready : OaEngineState::Failed;
	return status;
}

OaStatus OaComputeEngine::InitInPlaceImpl(const OaEngineConfig& InConfig) {
	// Initializes *this in place. `rt` aliases *this so the body reads as before.
	OaComputeEngine& rt = *this;

	OaValidation::InitFromEnv();
	// Translate OaNumericMode → env-knob state BEFORE any subsystem reads env.
	// User-supplied env vars still win (SetIfUnset checks first).
	OaApplyNumericMode(InConfig.NumericMode);

	if (InConfig.EnableMultiDevice) {
		// Multi-device path: mesh owns all Vulkan resources.
		// Engine.Device/Allocator/Pipelines/Bindless alias the Primary node.
		auto meshResult = OaDeviceMesh::Create(InConfig);
		if (!meshResult.IsOk()) return meshResult.GetStatus();

		auto mesh = OaMakeUniquePtr<OaDeviceMesh>(std::move(meshResult.GetValue()));
		if (!mesh->Primary) {
			return OaStatus::Error(OaStatusCode::DeviceNotFound, "mesh has no primary device");
		}

		// Shallow-copy primary node's resources into engine top-level fields.
		// These are the same Vulkan handles — mesh owns the actual resources.
		auto* pri = mesh->Primary;
		rt.Device = pri->Device;
		rt.Device.OwnsInstance = false;  // Engine's copy never destroys instance
		rt.Allocator = pri->Allocator;
		rt.Pipelines = std::move(pri->Pipelines);
		rt.Bindless = std::move(pri->Bindless);
		rt.Precision_ = InConfig.Precision;
		rt.MatrixPlacement_ = ResolveMatrixPlacement(rt.Device);

		if (rt.Precision_ == OaPrecision::BF16 && !rt.Device.NativeShaderBfloat16Usable()) {
			OA_LOG_WARN(OaLogComponent::Core, "BF16 requested but device lacks VK_KHR_shader_bfloat16 — falling back to FP32");
			rt.Precision_ = OaPrecision::FP32;
		}

		rt.Mesh_ = std::move(mesh);
		rt.LogSelectedDevices();
		rt.Context_.reset(OaContext::Create(&rt));

		OA_LOG_INFO(OaLogComponent::Core, "Multi-device: %u device(s)", rt.Mesh_->NodeCount());
		if (InConfig.EnableValidation) {
			OA_LOG_INFO(OaLogComponent::Core, "Validation layers: ON (VK_LAYER_KHRONOS_validation)");
		}
		if (InConfig.RegisterAsGlobal) {
			rt.SetAsGlobal();
		}

		// Register extension kernels into the dynamic table before pipeline compilation.
		{
			OaExtKernelRegistry extReg;
			for (OaExtension* ext : InConfig.Extensions) {
				ext->RegisterKernels(extReg);
			}
		}

		// Initialize shader provider (after external providers have been registered)
		OaShaderProviderInit(nullptr);

		// Pre-load all embedded shaders from the registry
		if (InConfig.PreloadEmbeddedPipelines) {
			auto loadStatus = rt.EnsureAllEmbeddedLiboaPipelines();
			if (!loadStatus.IsOk()) {
				OA_LOG_WARN(OaLogComponent::Core, "Failed to pre-load embedded pipelines: %s",
					loadStatus.GetMessage().c_str());
			}
		}

		return OaStatus::Ok();
	}

	// Single-device path
	OaDeviceType pref = OaDeviceType::VkDiscrete;
	switch (InConfig.DevicePref) {
		case OaDevicePreference::Integrated: pref = OaDeviceType::VkIntegrated; break;
		case OaDevicePreference::Cpu:        pref = OaDeviceType::VkCpu; break;
		default: break;
	}

	OaVec<const char*> instanceExtraPtrs;
	for (const auto& ext : InConfig.InstanceExtraExtensions) {
		if (!ext.Empty()) {
			instanceExtraPtrs.PushBack(ext.CStr());
		}
	}
	OaSpan<const char* const> extraSpan(instanceExtraPtrs.Data(), instanceExtraPtrs.Size());

	// Both Swapchain (GUI) and Headless modes need a graphics queue. Only
	// Swapchain triggers actual swapchain creation in OaGraphicsEngine —
	// Headless never attaches a surface. See Architecture/OaArchitecture.md §10.
	const OaBool wantPresentation =
		(InConfig.PresentationMode == OaPresentationMode::Swapchain
		 or InConfig.PresentationMode == OaPresentationMode::Headless);

	auto pickResult = InConfig.DevicePref == OaDevicePreference::ByIndex
		? OaVkDevice::Create(
			OaStringView(InConfig.AppName),
			InConfig.EnableValidation,
			pref,
			InConfig.DeviceIndex,
			InConfig.AppVersion,
			extraSpan,
			wantPresentation           // <-- new last arg
		) : OaVkDevice::Create(
			OaStringView(InConfig.AppName),
			InConfig.EnableValidation,
			pref,
			OaVkEnumerationIndexUnset,
			InConfig.AppVersion,
			extraSpan,
			wantPresentation           // <-- new last arg
		);

	if (!pickResult.IsOk()) {
		return pickResult.GetStatus();
	}

	auto allocator = OaVma::Create(pickResult.GetValue());
	if (!allocator.IsOk()) {
		pickResult.GetValue().Destroy();
		return allocator.GetStatus();
	}

	rt.Device = std::move(pickResult.GetValue());
	rt.Allocator = std::move(allocator.GetValue());
	rt.Precision_ = InConfig.Precision;
	rt.MatrixPlacement_ = ResolveMatrixPlacement(rt.Device);

	if (rt.Precision_ == OaPrecision::BF16 && !rt.Device.NativeShaderBfloat16Usable()) {
		OA_LOG_WARN(OaLogComponent::Core,
			"BF16 requested but device lacks VK_KHR_shader_bfloat16 — falling back to FP32");
		rt.Precision_ = OaPrecision::FP32;
	}

	rt.LogSelectedDevices();
	if (InConfig.EnableValidation) {
		OA_LOG_INFO(OaLogComponent::Core, "Validation layers: ON (VK_LAYER_KHRONOS_validation)");
	}

	auto bindlessResult = OaBindlessHeap::Create(rt.Device);
	if (!bindlessResult.IsOk()) {
		return OaStatus::Error(OaStatusCode::PipelineError,
			"Bindless heap creation failed (required for all operations)");
	}
	rt.Bindless = std::move(bindlessResult.GetValue());
	rt.Context_.reset(OaContext::Create(&rt));

	OaString cacheDir;
	if (InConfig.EnablePipelineCache) {
		cacheDir = InConfig.PipelineCacheDir;
	}
	OA_RETURN_IF_ERROR(rt.Pipelines.Init(rt.Device, cacheDir, rt.Bindless.PipelineLayout));

	if (InConfig.RegisterAsGlobal) {
		rt.SetAsGlobal();
	}

	// Register extension kernels into the dynamic table before pipeline compilation.
	{
		OaExtKernelRegistry extReg;
		for (OaExtension* ext : InConfig.Extensions) {
			ext->RegisterKernels(extReg);
		}
	}

	// Initialize shader provider (after external providers have been registered)
	OaShaderProviderInit(nullptr);

	// Pre-load all embedded shaders from the registry
	if (InConfig.PreloadEmbeddedPipelines) {
		auto loadStatus = rt.EnsureAllEmbeddedLiboaPipelines();
		if (!loadStatus.IsOk()) {
			OA_LOG_WARN(OaLogComponent::Core, "Failed to pre-load embedded pipelines: %s",
				loadStatus.GetMessage().c_str());
		}
	}

	// Initialize GEMM route cache
	rt.GemmRouteCache = new OaGemmRouteCache();
	
	// Load cached route policy from disk if available
	const char* cachePath = OaGemmRouteCache::DefaultPath;
	(void)rt.GemmRouteCache->Load(cachePath);

	return OaStatus::Ok();
}

void OaComputeEngine::Destroy() {
	// Guard against double-destroy (e.g. an explicit Destroy() followed by the
	// destructor). The engine is pinned, so there is no moved-from state to guard.
	if (State_ == OaEngineState::Destroying || State_ == OaEngineState::Destroyed) {
		return;
	}
	State_ = OaEngineState::Destroying;
	if (!Device.Device) {
		LifetimeToken_.Reset();
		State_ = OaEngineState::Destroyed;
		return;
	}
	// Expire buffer-owner weak references before any device/VMA state disappears.
	LifetimeToken_.Reset();
	ClearGlobal();
	if (OaContext::GetDefaultPtr() == Context_.get()) {
		OaContext::SetDefault(nullptr);
	}
	Context_.reset();
	ReleaseMeshDemoAuxBuffer();

	// Cleanup GEMM route cache
	if (GemmRouteCache) {
		const char* cachePath = OaGemmRouteCache::DefaultPath;
		(void)GemmRouteCache->Save(cachePath);
		delete GemmRouteCache;
		GemmRouteCache = nullptr;
	}

	if (ComputeBatchStream_) {
		OaVkBatch batch;
		batch.Stream = ComputeBatchStream_;
		ComputeBatchStream_ = nullptr;
		(void)OaVkDispatch::Flush(batch, *this);
	}

	// Sync and clear ring streams before pool destruction
	for (OaU32 i = 0; i < kBatchRingSize; ++i) {
		if (BatchRing_[i] and BatchRing_[i]->Submitted) {
			(void)BatchRing_[i]->Synchronize(Device);
		}
		BatchRing_[i] = nullptr;
	}
	BatchRingIdx_ = 0;

	// Destroy engine-level stream pools (always use primary device)
	for (auto& s : StreamPool_) s->Destroy(Device);
	StreamPool_.Clear();
	FreeStack_.Clear();
	for (auto& s : AsyncStreamPool_) s->Destroy(Device);
	AsyncStreamPool_.Clear();
	AsyncFreeStack_.Clear();
	TransferStream_.Destroy(Device);
	ReadbackStream_.Destroy(Device);
	Allocator.Free(ReadbackStaging_);
	UploadRing_.reset();

	{
		std::lock_guard<std::mutex> lock(HostVisibleBufferCacheMutex_);
		for (auto& buf : HostVisibleBufferCache_) {
			DeregisterBuffer(buf);
			Allocator.Free(buf);
		}
		HostVisibleBufferCache_.Clear();
		HostVisibleBufferCacheBytes_ = 0;
	}

	if (Mesh_) {
		// Multi-device: move pipelines/bindless back into primary node before mesh destroys them.
		// Engine took ownership of these from the primary node during Create().
		Mesh_->Primary->Pipelines = std::move(Pipelines);
		Mesh_->Primary->Bindless = std::move(Bindless);

		Mesh_->Destroy();
		Mesh_.reset();

		// Zero out engine aliases — handles already freed by mesh
		Device = OaVkDevice{};
		Allocator = OaVma{};
	} else {
		// Single-device: engine owns everything directly
		Pipelines.Destroy(Device);
		Bindless.Destroy(Device);
		Allocator.Destroy();
		Device.Destroy();
	}
	State_ = OaEngineState::Destroyed;
}

OaStatus OaComputeEngine::BeginComputeBatch() {
	if (ComputeBatchStream_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"BeginComputeBatch: already active");
	}
	OaU32 idx = BatchRingIdx_ % kBatchRingSize;
	if (!BatchRing_[idx]) {
		BatchRing_[idx] = AcquireStream();
		if (!BatchRing_[idx]) {
			return OaStatus::Error(OaStatusCode::VulkanError,
				"BeginComputeBatch: failed to acquire ring stream");
		}
	}
	// Begin() auto-waits if this slot was submitted (from kBatchRingSize steps ago).
	// In steady state this wait is instant: GPU finishes before CPU records the next step.
	OA_RETURN_IF_ERROR(BatchRing_[idx]->Begin(Device));
	ComputeBatchStream_ = BatchRing_[idx];
	return OaStatus::Ok();
}

OaStatus OaComputeEngine::FlushComputeBatch() {
	if (!ComputeBatchStream_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"FlushComputeBatch: no active batch");
	}
	OaVkStream* current = ComputeBatchStream_;
	ComputeBatchStream_ = nullptr;
	// Secondary graphs in a context-owned batch deliberately omit their own
	// compute -> host barriers. Emit exactly one visibility edge at the real
	// submission/readback boundary.
	current->RecordHostReadbackBarrier();

	// Submit with a GPU-side dependency on the previous ring slot.
	// This ensures the previous step's optimizer writes are visible before the
	// next step's forward pass executes (Vulkan does not provide implicit ordering
	// between separate vkQueueSubmit calls even to the same queue).
	OaStatus status;
	if (BatchRingIdx_ > 0) {
		OaU32 prevIdx = (BatchRingIdx_ - 1) % kBatchRingSize;
		if (BatchRing_[prevIdx] and BatchRing_[prevIdx]->Submitted) {
			status = current->SubmitWithDependency(
				*this,
				BatchRing_[prevIdx]->TimelineSem,
				BatchRing_[prevIdx]->TimelineValue);
		} else {
			status = current->Submit(*this);
		}
	} else {
		status = current->Submit(*this);
	}

	if (status.IsOk()) { BatchRingIdx_++; }
	return status;
}

OaStatus OaComputeEngine::SyncCurrentBatch() {
	if (BatchRingIdx_ == 0) { return OaStatus::Ok(); }
	OaU32 prevIdx = (BatchRingIdx_ - 1) % kBatchRingSize;
	if (!BatchRing_[prevIdx] or !BatchRing_[prevIdx]->Submitted) { return OaStatus::Ok(); }
	return BatchRing_[prevIdx]->Synchronize(Device);
}

OaCompletionToken OaComputeEngine::LastComputeBatchCompletion() const
{
	if (BatchRingIdx_ == 0U) return {};
	const OaU32 index = (BatchRingIdx_ - 1U) % kBatchRingSize;
	return BatchRing_[index]
		? BatchRing_[index]->Completion(Device)
		: OaCompletionToken();
}

OaStatus OaComputeEngine::EnsurePipeline(
	OaStringView InName,
	OaSpan<const OaU8> InSpirv,
	const OaPipelineSpec& InSpec)
{
	return Pipelines.EnsurePipeline(Device, InName, InSpirv, InSpec);
}

static void OaLoadSpvDirIntoPipelines(
	OaComputeEngine& InRt, const OaString& InDir, OaU32* InOutLoaded, OaU32* InOutSkipped)
{
	if (!InOutLoaded || !InOutSkipped) return;
	OaString spirvDir = InDir;
	if (spirvDir.empty()) return;
	if (spirvDir.back() != '/') spirvDir += '/';

	auto filesResult = OaFileIo::Glob(OaPath(spirvDir), "*.spv");
	if (!filesResult) return;

	for (const auto& path : filesResult.GetValue()) {
		OaString name = OaFileIo::GetStem(path);
		// Skip cooperative-matrix kernels — they are loaded (and cap-gated) only via
		// the embedded path. Match the real kernel names (GemmCmSgBf16 / GemmCmWgBf16,
		// "Cm" = CoopMat), not just a "CoopMat" substring the renamed kernels lack.
		if (name.find("CoopMat") != OaString::npos ||
			name.find("CmSg") != OaString::npos ||
			name.find("CmWg") != OaString::npos ||
			!OaComputeKernelUsesDefaultBindlessPipeline(name.c_str())) {
			++*InOutSkipped;
			continue;
		}
		auto data = OaFileIo::ReadBinary(path);
		if (!data || data.GetValue().Empty()) continue;
		const auto& bytes = data.GetValue();
		OaPipelineSpec spec{.WgSize = 256, .NumBindings = 16, .PushConstantBytes = 128,
			.SpecConstants = {{.Id = 0, .Value = InRt.DtypeSpecConstant()}}};
		if (InRt.EnsurePipeline(name, OaSpan<const OaU8>(bytes.Data(), bytes.Size()), spec).IsOk()) {
			++*InOutLoaded;
		}
	}
}

OaStatus OaComputeEngine::EnsureAllEmbeddedLiboaPipelines() {
	OaU32 loaded = 0;
	OaU32 skipped = 0;
	OaU32 failed = 0;
	const OaU32 total = OaSpvCount();

	class OaShaderLoadPlan {
	public:
		const char* Name = nullptr;
		OaU32 FirstRequest = 0;
		OaU32 RequestCount = 0;
	};
	OaVec<OaPipelineLoadRequest> requests;
	OaVec<OaShaderLoadPlan> plans;
	requests.Reserve(total * (Device.NativeShaderBfloat16Usable() ? 2u : 1u));
	plans.Reserve(total);

	for (OaU32 idx = 0; idx < total; ++idx) {
		const OaSpvEntry* ent = OaSpvFindByIndex(idx);
		if (!ent || !ent->Name || ent->Size == 0 || !ent->Data) continue;

		if (!OaComputeKernelUsesDefaultBindlessPipeline(ent->Name)) {
			++skipped;
			continue;
		}

		OaString nameStr(ent->Name);
		if (nameStr == "GemmCoopVec") {
			if (!Device.Info.Software.HasCooperativeVector) {
				OA_LOG_DEBUG(OaLogComponent::Core,
					"Skipping GemmCoopVec (VK_NV_cooperative_vector unavailable)");
				++skipped;
				continue;
			}
		}
		// Feature-gate ANY kernel that declares required device caps in the matmul
		// registry — cooperative-matrix, workgroup scope, bf16 input, etc. Gate by the
		// kernel's REAL registry name, NOT a "CoopMat" substring: the KHR kernels are
		// named GemmCmSgBf16 / GemmCmWgBf16 (and fused ...CmSgBf16 variants) — "Cm" =
		// CoopMat, with no "CoopMat" literal — so the old substring test never fired
		// for them and let a bf16 workgroup-CoopMat pipeline reach
		// vkCreateComputePipelines on a device (e.g. Intel pre-Xe2 Tiger Lake Xe on
		// Mesa/ANV) that enumerates the shape but crashes compiling it. GemmCapsMask()
		// is already vendor-trust-corrected for lying drivers, so an unsupported
		// variant is skipped cleanly and routing falls back to the fp32 tiled/naive
		// kernels (which always satisfy their caps).
		{
			const OaU64 required = OaMatmulRegistry::RequiredCapsMaskForShaderName(ent->Name);
			if (required != 0 && !OaMatmulRegistry::CapsSatisfy(GemmCapsMask(), required)) {
				OA_LOG_DEBUG(OaLogComponent::Core,
					"Skipping %s (required GEMM caps unavailable on this device)", ent->Name);
				++skipped;
				continue;
			}
			// Conservative fallback: a CoopMat/CmSg/CmWg-named kernel NOT in the
			// registry stays off unless the device broadly advertises coop-matrix.
			if (required == 0 &&
			    (nameStr.find("CoopMat") != OaString::npos ||
			     nameStr.find("CmSg") != OaString::npos ||
			     nameStr.find("CmWg") != OaString::npos) &&
			    !Device.Info.Software.HasCooperativeMatrix) {
				++skipped;
				continue;
			}
		}

		// Determine workgroup size based on kernel name
		OaU32 wgSize = 256;  // Default for most kernels
		if (nameStr.find("CoopMat") != OaString::npos) {
			wgSize = 128;  // CooperativeMatrix kernels use 128
		}

		// Register BOTH storage variants (DTYPE=0 FP32, DTYPE=1 BF16) so a dispatch can
		// select the one matching its operand tensors (OaComputeNode::Dtype), instead of a
		// global engine mode. DTYPE=1 is skipped when the device can't do native bf16. In
		// FP32 engine mode every tensor is FP32 → node dtype 0 → the =0 variant (unchanged);
		// only the extra =1 variants are loaded (unused). See OaPrecisionDtype.md.
		const OaU32 maxDtype = Device.NativeShaderBfloat16Usable() ? 1u : 0u;
		OaShaderLoadPlan plan;
		plan.Name = ent->Name;
		plan.FirstRequest = static_cast<OaU32>(requests.Size());
		plan.RequestCount = maxDtype + 1u;
		for (OaU32 dt = 0; dt <= maxDtype; ++dt) {
			OaPipelineSpec spec{.WgSize = wgSize, .NumBindings = 16, .PushConstantBytes = 128,
				.SpecConstants = {{.Id = 0, .Value = dt}}};
			requests.PushBack(OaPipelineLoadRequest{
				.Name = ent->Name,
				.Spirv = OaSpan<const OaU8>(ent->Data, ent->Size),
				.Spec = std::move(spec),
			});
		}
		plans.PushBack(plan);
	}

	const OaI64 configuredThreads = OaEnvFlag::GetInt("OA_SHADER_LOAD_THREADS", 0);
	const OaU32 hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
	// A populated Vulkan pipeline cache makes pipeline creation extremely cheap;
	// cloning and merging it per worker costs more than serial creation. Cold
	// driver compilation is CPU-heavy and scales best around physical-core count,
	// approximated portably as half of logical CPUs and capped to avoid excessive
	// cache replication on large hosts. The environment override remains exact.
	OaU32 loadThreads = 1;
	if (configuredThreads > 0) {
		loadThreads = static_cast<OaU32>(std::min<OaI64>(configuredThreads, 64));
	} else if (!Pipelines.HasInitialCacheData()) {
		loadThreads = std::min<OaU32>(std::max<OaU32>(1u, hardwareThreads / 2u), 8u);
	}
	loadThreads = std::max<OaU32>(1u,
		std::min<OaU32>(loadThreads, static_cast<OaU32>(requests.Size())));

	OA_LOG_INFO(OaLogComponent::Core,
		"Preloading %zu shader pipelines (%u thread%s, %s cache)",
		requests.Size(), loadThreads, loadThreads == 1 ? "" : "s",
		Pipelines.HasInitialCacheData() ? "warm" : "cold");

	const auto loadBegin = std::chrono::steady_clock::now();
	OaVec<OaStatus> requestStatuses;
	if (!requests.Empty()) {
		(void)Pipelines.EnsurePipelinesParallel(
			Device,
			OaSpan<const OaPipelineLoadRequest>(requests.Data(), requests.Size()),
			loadThreads,
			&requestStatuses);
	}
	const OaF64 loadMs = std::chrono::duration<OaF64, std::milli>(
		std::chrono::steady_clock::now() - loadBegin).count();

	for (const auto& plan : plans) {
		OaStatus shaderStatus = OaStatus::Ok();
		for (OaU32 offset = 0; offset < plan.RequestCount; ++offset) {
			const OaStatus& status = requestStatuses[plan.FirstRequest + offset];
			if (status.IsError()) shaderStatus = status;
		}
		if (shaderStatus.IsOk()) {
			++loaded;
		} else {
			++failed;
			OA_LOG_WARN(OaLogComponent::Core, "Failed to load shader '%s': %s",
				plan.Name, shaderStatus.GetMessage().c_str());
		}
	}

	if (loaded == 0) {
		OaU32 fromPaths = 0;
		OaU32 pathSkipped = 0;
		for (const OaString& dir : ShaderSearchPaths_) {
			OaLoadSpvDirIntoPipelines(*this, dir, &fromPaths, &pathSkipped);
		}
		skipped += pathSkipped;
		loaded = fromPaths;
	}

	OA_LOG_INFO(OaLogComponent::Core,
		"Loaded %u/%u shaders (skipped=%u, failed=%u, pipelines=%zu, threads=%u, %.2f ms, precision=%s)",
		loaded, total, skipped, failed, requests.Size(), loadThreads, loadMs,
		GetPrecision() == OaPrecision::BF16 ? "BF16" : "FP32");

	if (loaded == 0) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"no liboa compute pipelines loaded (embed empty and no *.spv under AddShaderSearchPath); "
			"use release liboa with embedded shaders or add a spirv search path before Init");
	}
	return OaStatus::Ok();
}

OaComputePipeline& OaComputeEngine::GetPipeline(OaStringView InName) {
	return Pipelines.GetPipeline(InName, DtypeSpecConstant());
}

OaComputePipeline& OaComputeEngine::GetPipeline(OaKernelId InKernelId) {
	const OaComputeKernel* kernel = OaComputeKernelFindByPackedId(InKernelId);
	if (!kernel) {
		OA_LOG_WARN(OaLogComponent::Core, "GetPipeline: unknown kernel id=0x%llx",
			static_cast<unsigned long long>(InKernelId));
		return Pipelines.GetPipeline("", DtypeSpecConstant());
	}
	return Pipelines.GetPipeline(kernel->Name, DtypeSpecConstant());
}

void OaComputeEngine::AddShaderSearchPath(const OaString& InPath) {
	OaString path = InPath;
	if (!path.empty() && path.back() != '/') path += '/';
	ShaderSearchPaths_.PushBack(std::move(path));
}

// ─── Buffer Allocation ─────────────────────────────────────────────────────

OaResult<OaVkBuffer> OaComputeEngine::AllocBuffer(OaU64 InSize) {
	return AllocBuffer(InSize, OaMemoryPlacement::HostUpload);
}

OaResult<OaVkBuffer> OaComputeEngine::AllocBuffer(
	OaU64 InSize, OaMemoryPlacement InPlacement)
{
	const OaMemoryPlacement placement = InPlacement == OaMemoryPlacement::Auto
		? MatrixPlacement_
		: InPlacement;
	if (placement == OaMemoryPlacement::DeviceLocal) {
		return AllocBufferDevice(InSize);
	}
	if (placement == OaMemoryPlacement::Unified) {
		return AllocBufferBar(InSize);
	}
	if (placement == OaMemoryPlacement::HostReadback) {
		auto result = Allocator.AllocHostReadback(InSize);
		if (result) RegisterBuffer(*result);
		return result;
	}

	{
		std::lock_guard<std::mutex> lock(HostVisibleBufferCacheMutex_);
		OaI32 best = -1;
		OaU64 bestSize = UINT64_MAX;
		for (OaU32 i = 0; i < HostVisibleBufferCache_.Size(); ++i) {
			const auto& candidate = HostVisibleBufferCache_[i];
			const OaU64 capacity = candidate.Capacity != 0 ? candidate.Capacity : candidate.Size;
			if (capacity >= InSize && capacity < bestSize) {
				best = static_cast<OaI32>(i);
				bestSize = capacity;
			}
		}
		if (best >= 0) {
			OaVkBuffer reused = HostVisibleBufferCache_[static_cast<OaU32>(best)];
			HostVisibleBufferCache_.Erase(HostVisibleBufferCache_.begin() + best);
			HostVisibleBufferCacheBytes_ -= reused.Capacity;
			reused.Size = InSize;
			return reused;
		}
	}

	auto res = Allocator.AllocHostVisible(InSize);
	if (!res) {
		return res;
	}
	RegisterBuffer(*res);
	return res;
}

OaStatus OaComputeEngine::UploadBuffer(
	const OaVkBuffer& InDst, OaU64 InDstOffset, const void* InData, OaU64 InSize)
{
	if (!InDst.Buffer || !InData || InSize == 0 || InDstOffset > InDst.Size
		|| InSize > InDst.Size - InDstOffset) {
		return OaStatus::InvalidArgument("UploadBuffer: invalid source or destination range");
	}
	if (InDst.MappedPtr) {
		OaMemcpy(static_cast<OaU8*>(InDst.MappedPtr) + InDstOffset, InData,
			static_cast<OaUsize>(InSize));
		return Allocator.FlushHostBuffer(InDst, InDstOffset, InSize)
			? OaStatus::Ok()
			: OaStatus::Error(OaStatusCode::VulkanError, "UploadBuffer: mapped flush failed");
	}

	// vkCmdCopyBuffer requires four-byte aligned offsets and sizes. Matrices may
	// still contain byte/half scalars or expose an unaligned view, so promote the
	// transfer to the enclosing words. Partial updates preserve neighbouring
	// bytes through a read-modify-write; whole-buffer uploads simply zero pad the
	// physical tail allocated by OaVma.
	OaVkBuffer copyDst = InDst;
	OaU64 copyOffset = InDstOffset;
	const void* copyData = InData;
	OaU64 copySize = InSize;
	OaVec<OaU8> alignedData;
	if ((copyOffset & 3ULL) != 0 || (copySize & 3ULL) != 0) {
		const OaU64 alignedBegin = copyOffset & ~3ULL;
		const OaU64 alignedEnd = (copyOffset + copySize + 3ULL) & ~3ULL;
		if (alignedEnd > copyDst.Capacity) {
			return OaStatus::InvalidArgument("UploadBuffer: padded range exceeds capacity");
		}
		alignedData.Resize(static_cast<OaUsize>(alignedEnd - alignedBegin));
		if (copyOffset != 0 || copySize != InDst.Size) {
			OaVkBuffer physical = InDst;
			physical.Size = physical.Capacity;
			OA_RETURN_IF_ERROR(ReadbackBuffer(physical, alignedBegin,
				alignedData.Data(), alignedEnd - alignedBegin));
		} else {
			OaMemzero(alignedData.Data(), alignedData.Size());
		}
		OaMemcpy(alignedData.Data() + (copyOffset - alignedBegin), copyData,
			static_cast<OaUsize>(copySize));
		copyDst.Size = copyDst.Capacity;
		copyOffset = alignedBegin;
		copyData = alignedData.Data();
		copySize = alignedEnd - alignedBegin;
	}

	std::lock_guard<std::mutex> lock(UploadRingMutex_);
	if (!UploadRing_ || UploadRing_->FrameCapacityBytes() < copySize) {
		UploadRing_.reset();
		const OaU64 frameBytes = (copySize + 255ULL) & ~255ULL;
		const OaU64 capacity = std::max<OaU64>(64ULL * 1024ULL * 1024ULL, frameBytes * 3ULL);
		auto ring = OaUploadRing::Create(*this, OaUploadRingConfig{
			.CapacityBytes = capacity,
			.FramesInFlight = 3,
			.Alignment = 256,
		});
		if (!ring) return ring.GetStatus();
		UploadRing_ = OaMakeUniquePtr<OaUploadRing>(std::move(*ring));
	}
	OA_RETURN_IF_ERROR(UploadRing_->BeginBatch());
	OA_RETURN_IF_ERROR(UploadRing_->Upload(
		copyDst, copyOffset, copyData, copySize));
	auto completion = UploadRing_->Submit();
	if (!completion) return completion.GetStatus();
	return completion->Wait();
}

OaStatus OaComputeEngine::ReadbackBuffer(
	const OaVkBuffer& InSrc, OaU64 InSrcOffset, void* OutData, OaU64 InSize)
{
	if (!InSrc.Buffer || !OutData || InSize == 0 || InSrcOffset > InSrc.Size
		|| InSize > InSrc.Size - InSrcOffset) {
		return OaStatus::InvalidArgument("ReadbackBuffer: invalid source or destination range");
	}
	if (InSrc.MappedPtr) {
		if (!Allocator.InvalidateHostBuffer(InSrc, InSrcOffset, InSize)) {
			return OaStatus::Error(OaStatusCode::VulkanError, "ReadbackBuffer: mapped invalidate failed");
		}
		OaMemcpy(OutData, static_cast<const OaU8*>(InSrc.MappedPtr) + InSrcOffset,
			static_cast<OaUsize>(InSize));
		return OaStatus::Ok();
	}
	const OaU64 copyOffset = InSrcOffset & ~3ULL;
	const OaU64 copyEnd = (InSrcOffset + InSize + 3ULL) & ~3ULL;
	if (copyEnd > InSrc.Capacity) {
		return OaStatus::InvalidArgument("ReadbackBuffer: padded range exceeds capacity");
	}
	const OaU64 copySize = copyEnd - copyOffset;

	std::lock_guard<std::mutex> lock(ReadbackMutex_);
	if (!ReadbackStream_.CommandPool) {
		auto streamResult = OaVkStream::Create(
			Device, Device.Queues.ComputeQueueFamily, Device.Queues.ComputeQueue);
		if (!streamResult) return streamResult.GetStatus();
		ReadbackStream_ = std::move(*streamResult);
	}
	if (!ReadbackStaging_.Buffer || ReadbackStaging_.Capacity < copySize) {
		// Begin() below would also wait before command-buffer reuse, but an old
		// staging allocation cannot be released until its previous copy completes.
		if (ReadbackStream_.Submitted) {
			OA_RETURN_IF_ERROR(ReadbackStream_.Synchronize(Device));
		}
		Allocator.Free(ReadbackStaging_);
		OaU64 capacity = 64ULL * 1024ULL;
		while (capacity < copySize && capacity <= UINT64_MAX / 2ULL) capacity *= 2ULL;
		if (capacity < copySize) capacity = copySize;
		auto readbackResult = Allocator.AllocHostReadback(capacity);
		if (!readbackResult) return readbackResult.GetStatus();
		ReadbackStaging_ = std::move(*readbackResult);
	}

	OaStatus status = ReadbackStream_.Begin(Device);
	if (status.IsOk()) {
		const OaBufferCopyRegion region{
			.SrcOffset = copyOffset,
			.DstOffset = 0,
			.Size = copySize,
		};
		ReadbackStream_.RecordCopyBufferRegions(InSrc, ReadbackStaging_,
			OaSpan<const OaBufferCopyRegion>(&region, 1));
		status = ReadbackStream_.Submit(*this);
	}
	if (status.IsOk()) status = ReadbackStream_.Synchronize(Device);
	if (status.IsOk() && !Allocator.InvalidateHostBuffer(ReadbackStaging_, 0, copySize)) {
		status = OaStatus::Error(OaStatusCode::VulkanError, "ReadbackBuffer: staging invalidate failed");
	}
	if (status.IsOk()) {
		OaMemcpy(OutData,
			static_cast<const OaU8*>(ReadbackStaging_.MappedPtr) + (InSrcOffset - copyOffset),
			static_cast<OaUsize>(InSize));
	}
	return status;
}

OaResult<OaVkBuffer> OaComputeEngine::AllocBufferDevice(OaU64 InSize) {
	auto res = Allocator.AllocDevice(InSize);
	if (!res) {
		return res;
	}
	RegisterBuffer(*res);
	return res;
}

OaResult<OaVkBuffer> OaComputeEngine::AllocBufferBar(OaU64 InSize) {
	auto res = Allocator.AllocBar(InSize);
	if (!res) {
		return res;
	}
	RegisterBuffer(*res);
	return res;
}

void OaComputeEngine::FreeBuffer(OaVkBuffer& InOutBuffer) {
	if (InOutBuffer.Buffer && InOutBuffer.MappedPtr
		&& InOutBuffer.Placement == OaMemoryPlacement::HostUpload
		&& (!Mesh_ || InOutBuffer.NodeIndex == 0) &&
		!InOutBuffer.IsBar() && !InOutBuffer.IsImported()
		&& !InOutBuffer.IsTransient() &&
		InOutBuffer.BindlessIndex != OA_BINDLESS_INVALID) {
		std::lock_guard<std::mutex> lock(HostVisibleBufferCacheMutex_);
		const OaBool canCache =
			HostVisibleBufferCache_.Size() < kHostVisibleCacheMaxBuffers &&
			InOutBuffer.Capacity <= kHostVisibleCacheMaxBytes &&
			HostVisibleBufferCacheBytes_ + InOutBuffer.Capacity <= kHostVisibleCacheMaxBytes;
		if (canCache) {
			HostVisibleBufferCacheBytes_ += InOutBuffer.Capacity;
			HostVisibleBufferCache_.PushBack(InOutBuffer);
			InOutBuffer = OaVkBuffer{};
			return;
		}
	}
	DeregisterBuffer(InOutBuffer);
	Allocator.Free(InOutBuffer);
}

// ─── Bindless ──────────────────────────────────────────────────────────────

OaU32 OaComputeEngine::RegisterBuffer(OaVkBuffer& InOutBuffer) {
	InOutBuffer.NodeIndex = 0;
	if (!Bindless.DescriptorSet) {
		return OA_BINDLESS_INVALID;
	}
	OaU32 idx = Bindless.Register(Device, InOutBuffer);
	InOutBuffer.BindlessIndex = idx;
	return idx;
}

OaU32 OaComputeEngine::RegisterBufferForOwnedNode(OaVkBuffer& InOutBuffer) {
	if (InOutBuffer.NodeIndex == 0 || !Mesh_) {
		return RegisterBuffer(InOutBuffer);
	}
	auto* node = GetNode(InOutBuffer.NodeIndex);
	if (!node) {
		return OA_BINDLESS_INVALID;
	}
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	OaU32 idx = RegisterBufferOnMeshNode(*node, InOutBuffer);
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
	return idx;
}

void OaComputeEngine::DeregisterBuffer(OaVkBuffer& InOutBuffer) {
	if (InOutBuffer.BindlessIndex == OA_BINDLESS_INVALID) {
		return;
	}
	if (InOutBuffer.NodeIndex == 0 || !Mesh_) {
		Bindless.Deregister(InOutBuffer.BindlessIndex);
		InOutBuffer.BindlessIndex = OA_BINDLESS_INVALID;
		return;
	}
	auto* node = GetNode(InOutBuffer.NodeIndex);
	if (!node) {
		InOutBuffer.BindlessIndex = OA_BINDLESS_INVALID;
		return;
	}
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	node->Bindless.Deregister(InOutBuffer.BindlessIndex);
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
	InOutBuffer.BindlessIndex = OA_BINDLESS_INVALID;
}

// ─── Stream Pool ───────────────────────────────────────────────────────────

OaVkStream* OaComputeEngine::AcquireStream() {
	OaSpinlockGuard guard(StreamPoolLock_);
	if (!FreeStack_.Empty()) {
		OaU32 idx = FreeStack_.Back();
		FreeStack_.PopBack();
		return StreamPool_[idx].get();
	}
	auto res = OaVkStream::CreateCompute(Device);
	if (!res) {
		return nullptr;
	}
	auto ptr = OaMakeUniquePtr<OaVkStream>(std::move(*res));
	OaVkStream* raw = ptr.get();
	StreamPool_.PushBack(std::move(ptr));
	return raw;
}

void OaComputeEngine::ReleaseStream(OaVkStream* InStream) {
	OaSpinlockGuard guard(StreamPoolLock_);
	for (OaU32 i = 0; i < StreamPool_.Size(); ++i) {
		if (StreamPool_[i].get() == InStream) {
			FreeStack_.PushBack(i);
			return;
		}
	}
}

// ─── Async Compute Stream Pool ─────────────────────────────────────────────

OaVkStream* OaComputeEngine::AcquireAsyncStream() {
	if (!Device.Queues.HasAsyncCompute) {
		return AcquireStream();
	}
	OaSpinlockGuard guard(AsyncStreamPoolLock_);
	if (!AsyncFreeStack_.Empty()) {
		OaU32 idx = AsyncFreeStack_.Back();
		AsyncFreeStack_.PopBack();
		return AsyncStreamPool_[idx].get();
	}
	auto res = OaVkStream::Create(
		Device, Device.Queues.AsyncComputeQueueFamily, Device.Queues.AsyncComputeQueue);
	if (!res) {
		return nullptr;
	}
	auto ptr = OaMakeUniquePtr<OaVkStream>(std::move(*res));
	OaVkStream* raw = ptr.get();
	AsyncStreamPool_.PushBack(std::move(ptr));
	return raw;
}

void OaComputeEngine::ReleaseAsyncStream(OaVkStream* InStream) {
	if (!Device.Queues.HasAsyncCompute) { ReleaseStream(InStream); return; }
	OaSpinlockGuard guard(AsyncStreamPoolLock_);
	for (OaU32 i = 0; i < AsyncStreamPool_.Size(); ++i) {
		if (AsyncStreamPool_[i].get() == InStream) {
			AsyncFreeStack_.PushBack(i);
			return;
		}
	}
}

// ─── Thread-Safe Queue Submit ──────────────────────────────────────────────

OaStatus OaComputeEngine::SubmitToQueue(void* InQueue, void* InSubmitInfo, void* InFence) {
	VkQueue queue = static_cast<VkQueue>(InQueue);
	VkFence fence = static_cast<VkFence>(InFence);
	const VkSubmitInfo* si = static_cast<const VkSubmitInfo*>(InSubmitInfo);

	if (InQueue == Device.Queues.ComputeQueue) {
		std::lock_guard<std::mutex> lock(ComputeQueueMutex_);
		VkResult r = vkQueueSubmit(queue, 1, si, fence);
		if (r != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "vkQueueSubmit (compute) failed, VkResult=%d", static_cast<int>(r));
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (compute) failed");
		}
	} else if (Device.Queues.HasAsyncCompute && InQueue == Device.Queues.AsyncComputeQueue) {
		std::lock_guard<std::mutex> lock(AsyncComputeQueueMutex_);
		VkResult r = vkQueueSubmit(queue, 1, si, fence);
		if (r != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "vkQueueSubmit (async compute) failed, VkResult=%d", static_cast<int>(r));
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (async compute) failed");
		}
	} else if (InQueue == Device.Queues.TransferQueue) {
		std::lock_guard<std::mutex> lock(TransferQueueMutex_);
		VkResult r = vkQueueSubmit(queue, 1, si, fence);
		if (r != VK_SUCCESS) {
			OA_LOG_ERROR(OaLogComponent::Core, "vkQueueSubmit (transfer) failed, VkResult=%d", static_cast<int>(r));
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (transfer) failed");
		}
	} else {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "unknown queue");
	}
	return OaStatus::Ok();
}

OaStatus OaComputeEngine::SubmitToQueue2(void* InQueue, const VkSubmitInfo2* InSubmitInfo) {
	VkQueue queue = static_cast<VkQueue>(InQueue);

	if (InQueue == Device.Queues.ComputeQueue) {
		std::lock_guard<std::mutex> lock(ComputeQueueMutex_);
		VkResult r = vkQueueSubmit2(queue, 1, InSubmitInfo, VK_NULL_HANDLE);
		if (r != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit2 (compute) failed");
		}
	} else if (Device.Queues.HasAsyncCompute and InQueue == Device.Queues.AsyncComputeQueue) {
		std::lock_guard<std::mutex> lock(AsyncComputeQueueMutex_);
		VkResult r = vkQueueSubmit2(queue, 1, InSubmitInfo, VK_NULL_HANDLE);
		if (r != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit2 (async compute) failed");
		}
	} else if (InQueue == Device.Queues.TransferQueue) {
		std::lock_guard<std::mutex> lock(TransferQueueMutex_);
		VkResult r = vkQueueSubmit2(queue, 1, InSubmitInfo, VK_NULL_HANDLE);
		if (r != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit2 (transfer) failed");
		}
	} else {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "unknown queue");
	}
	return OaStatus::Ok();
}

// ─── Async Transfer ────────────────────────────────────────────────────────

OaStatus OaComputeEngine::CopyBufferAsync(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize) {
	if (!TransferStream_.CommandPool) {
		// OA buffers use exclusive queue-family ownership. Until graph-level
		// release/acquire transfers exist, a distinct transfer family cannot safely
		// consume them. A second queue in the compute family is fine; otherwise use
		// the primary compute queue.
		const bool sharedFamily =
			Device.Queues.TransferQueueFamily == Device.Queues.ComputeQueueFamily;
		auto res = OaVkStream::Create(
			Device,
			sharedFamily ? Device.Queues.TransferQueueFamily : Device.Queues.ComputeQueueFamily,
			sharedFamily ? Device.Queues.TransferQueue : Device.Queues.ComputeQueue
		);
		if (!res) {
			return res.GetStatus();
		}
		TransferStream_ = std::move(*res);
	}
	OA_RETURN_IF_ERROR(TransferStream_.Begin(Device));
	TransferStream_.RecordCopyBuffer(InSrc, InDst, InSize);
	return TransferStream_.Submit(*this);
}

OaStatus OaComputeEngine::WaitTransfer() {
	return TransferStream_.Synchronize(Device);
}

// ─── Multi-Device Mesh ─────────────────────────────────────────────────────

OaU32 OaComputeEngine::DeviceCount() const {
	return Mesh_ ? Mesh_->NodeCount() : 1;
}

OaDeviceNode* OaComputeEngine::GetNode(OaU32 InIndex) {
	return Mesh_ ? Mesh_->GetNode(InIndex) : nullptr;
}

OaDeviceNode* OaComputeEngine::GetPrimary() {
	return Mesh_ ? Mesh_->Primary : nullptr;
}

OaDeviceNode* OaComputeEngine::GetAuxiliary() {
	return Mesh_ ? Mesh_->GetByRole(OaDeviceRole::Auxiliary) : nullptr;
}

bool OaComputeEngine::IsMultiDevice() const {
	return Mesh_ && Mesh_->NodeCount() > 1;
}

OaVkStream* OaComputeEngine::AcquireStreamOn(OaU32 InNodeIndex) {
	if (!Mesh_) {
		return (InNodeIndex == 0) ? AcquireStream() : nullptr;
	}
	auto* node = Mesh_->GetNode(InNodeIndex);
	if (!node) {
		return nullptr;
	}
	return node->AcquireStream();
}

void OaComputeEngine::ReleaseStreamOn(OaU32 InNodeIndex, OaVkStream* InStream) {
	if (!Mesh_) {
		if (InNodeIndex == 0) {
			ReleaseStream(InStream);
			return;
		}
	}
	auto* node = Mesh_->GetNode(InNodeIndex);
	if (node) {
		node->ReleaseStream(InStream);
	}
}

OaStatus OaComputeEngine::SubmitToNodeQueue(
	OaU32 InNodeIndex, void* InQueue, void* InSubmitInfo, void* InFence,
	OaBool InDispatchAlreadyLoadedForNode
) {
	if (!Mesh_ || InNodeIndex == 0) {
		return SubmitToQueue(InQueue, InSubmitInfo, InFence);
	}

	auto* node = Mesh_->GetNode(InNodeIndex);
	if (!node) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "invalid node index");
	}

	VkQueue queue = static_cast<VkQueue>(InQueue);
	VkFence fence = static_cast<VkFence>(InFence);
	const VkSubmitInfo* si = static_cast<const VkSubmitInfo*>(InSubmitInfo);

	if (InDispatchAlreadyLoadedForNode) {
		VkResult r = vkQueueSubmit(queue, 1, si, fence);
		if (r != VK_SUCCESS) {
			return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (node) failed");
		}
		return OaStatus::Ok();
	}

	// Non-primary nodes: serialize through DeviceLoadLock since we need to
	// temporarily switch global function pointers.
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));

	VkResult r = vkQueueSubmit(queue, 1, si, fence);

	// Restore primary device as global
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));

	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkQueueSubmit (node) failed");
	}
	return OaStatus::Ok();
}

// ─── Mesh demo (auxiliary node compute) ────────────────────────────────────

void OaComputeEngine::ReleaseMeshDemoAuxBuffer() {
	if (!MeshDemoAuxBuf_.Buffer) {
		return;
	}
	FreeBufferOnNode(MeshDemoAuxBuf_);
	MeshDemoAuxNode_ = 0;
	MeshDemoAuxScaleReady_ = false;
}

OaStatus OaComputeEngine::EnsurePipelineOnNode(
	OaU32 InNodeIndex,
	OaStringView InName,
	OaSpan<const OaU8> InSpirv,
	const OaPipelineSpec& InSpec){
	if (InNodeIndex == 0) {
		return EnsurePipeline(InName, InSpirv, InSpec);
	}
	auto* node = GetNode(InNodeIndex);
	if (!node) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "EnsurePipelineOnNode: invalid node");
	}
	if (!Mesh_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "EnsurePipelineOnNode: no mesh");
	}
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	OaStatus st = node->Pipelines.EnsurePipeline(node->Device, InName, InSpirv, InSpec);
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
	return st;
}

OaResult<OaVkBuffer> OaComputeEngine::AllocBufferOnNode(OaU32 InNodeIndex, OaU64 InSize) {
	if (InNodeIndex == 0) {
		return AllocBuffer(InSize);
	}
	auto* node = GetNode(InNodeIndex);
	if (!node) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AllocBufferOnNode: invalid node");
	}
	if (!Mesh_) {
		return OaStatus::Error(OaStatusCode::InvalidArgument, "AllocBufferOnNode: no mesh");
	}
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	auto res = node->Allocator.AllocHostVisible(InSize);
	if (!res) {
		OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
		return res;
	}
	(void)RegisterBufferOnMeshNode(*node, *res);
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
	return res;
}

void OaComputeEngine::FreeBufferOnNode(OaVkBuffer& InOutBuffer) {
	const OaU32 ni = InOutBuffer.NodeIndex;
	if (ni == 0) {
		FreeBuffer(InOutBuffer);
		return;
	}
	auto* node = GetNode(ni);
	if (!node) {
		InOutBuffer = {};
		return;
	}
	if (!Mesh_) {
		InOutBuffer = {};
		return;
	}
	OaSpinlockGuard guard(*Mesh_->DeviceLoadLock);
	OaVkLoadDevice(static_cast<VkDevice>(node->Device.Device));
	if (InOutBuffer.BindlessIndex != OA_BINDLESS_INVALID) {
		node->Bindless.Deregister(InOutBuffer.BindlessIndex);
		InOutBuffer.BindlessIndex = OA_BINDLESS_INVALID;
	}
	node->Allocator.Free(InOutBuffer);
	OaVkLoadDevice(static_cast<VkDevice>(Device.Device));
	InOutBuffer = {};
}

OaStatus OaComputeEngine::PulseAuxiliaryMeshDemoCompute() {
	if (!IsMultiDevice()) {
		return OaStatus::Ok();
	}
	auto* aux = GetAuxiliary();
	if (!aux) {
		return OaStatus::Ok();
	}
	const OaU32 auxIdx = aux->Index;

	constexpr OaU32 DemoElems = 1024u;
	const OaU64 demoBytes = static_cast<OaU64>(DemoElems) * sizeof(OaF32);

	if (!MeshDemoAuxBuf_.Buffer) {
		auto buf = AllocBufferOnNode(auxIdx, demoBytes);
		if (!buf) {
			return buf.GetStatus();
		}
		MeshDemoAuxBuf_ = std::move(*buf);
		MeshDemoAuxNode_ = auxIdx;
	}

	if (!MeshDemoAuxScaleReady_) {
		const OaSpvEntry* spv = OaSpvFindAny("Scale");
		if (!spv || spv->Size == 0) {
			return OaStatus::NotFound("mesh demo: embedded `scale` SPIR-V not found");
		}
		const OaPrecisionTag tag = OaLookupPrecision("Scale");
		const OaU32 dtype = OaPrecisionDtype(tag, GetPrecision());
		const OaPipelineSpec spec{.WgSize = 256,
			.NumBindings = 2,
			.PushConstantBytes = 8,
			.SpecConstants = {{.Id = 0, .Value = dtype}}};
		OA_RETURN_IF_ERROR(EnsurePipelineOnNode(auxIdx, "Scale",
			OaSpan<const OaU8>(spv->Data, spv->Size), spec));
		MeshDemoAuxScaleReady_ = true;
	}

	if (MeshDemoAuxBuf_.MappedPtr) {
		OaF32* p = static_cast<OaF32*>(MeshDemoAuxBuf_.MappedPtr);
		for (OaU32 i = 0; i < DemoElems; ++i) {
			p[i] = 1.0f;
		}
	}

	struct {
		OaU32 N;
		OaF32 Scale;
	} push{DemoElems, 2.0f};
	OaVkBuffer bufs[] = {MeshDemoAuxBuf_, MeshDemoAuxBuf_};

	OA_RETURN_IF_ERROR(OaVkDispatch::RunOn(*this, auxIdx, "Scale", bufs,
		&push, sizeof(push), OaDivCeil(DemoElems, 256u)));
	return OaStatus::Ok();
}
