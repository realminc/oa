// Runtime ownership:
//   OaEngine     — the one concrete Vulkan resource owner.
//   OaPresenter  — optional WSI/swapchain service borrowing an OaEngine.
//
// Two-phase render init (SDL3 example)
// ─────────────────────────────────────
//   // Phase A — instance + graphics-capable device, no surface yet:
//   OaEngineConfig cfg;
//   cfg.PresentationMode = OaPresentationMode::Swapchain;
//   win.GetPresenterInstanceExtensions(&exts);         // SDL3 fills this
//   for (auto e : exts) cfg.InstanceExtraExtensions.PushBack(e);
//   auto eng = OaEngine::Create(cfg).GetValue();
//   OaPresenter presenter(*eng);
//
//   // Phase B — caller creates surface against the live VkInstance:
//   VkSurfaceKHR surf = VK_NULL_HANDLE;
//   win.CreatePresenterVkSurface(
//       static_cast<VkInstance>(eng->Device.Instance), &surf);
//
//   // Phase C — attach surface, build swapchain:
//   int w, h;  SDL_GetWindowSizeInPixels(sdlWin, &w, &h);
//   presenter.InitPresentation(surf, { uint32_t(w), uint32_t(h) });
//
//   // Phase D — optional ImGui (compile with -DOA_IMGUI):
//   presenter.InitImGui(win.GetNativeWindowHandle());
//
// Per-frame:
//   presenter.BeginImGuiFrame();   // no-op without OA_IMGUI
//   // ImGui / imnodes calls
//   presenter.EndImGuiFrame();     // no-op without OA_IMGUI
//   presenter.DrawFrame();
//
// Window swap (e.g. splash → editor):
//   VkInstance inst = static_cast<VkInstance>(eng->Device.Instance);
//   presenter.DetachPresentation();                     // swapchain torn down
//   vkDestroySurfaceKHR(inst, oldSurf, nullptr);        // caller destroys surface
//   win.reset(); newWin = CreateSdl3(newCfg);
//   newWin.CreatePresenterVkSurface(inst, &newSurf);
//   presenter.InitPresentation(newSurf, { uint32_t(w), uint32_t(h) });

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Device.h>
#include <Oa/Core/Extension.h>
#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <Oa/Core/Thread.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/Bindless.h>
#include <Oa/Runtime/GemmRouteCache.h>
#include <Oa/Runtime/Swapchain.h>
#include <Oa/Runtime/Topology.h>

#include <vulkan/vulkan.h>


class OaVkDispatch;
class OaVkImageDispatchTicket;
class OaUploadRing;
class OaContext;
class OaExecutionSession;
class OaComputeGraph;
class OaExecutionPlan;
class OaBorrowedServiceRetirement;
class OaGraphicsStreamLease;
struct OaRetiredUploadRing;
struct OaRetiredPresenter;


// ─── Config ─────────────────────────────────────────────────────────────────

enum class OaDevicePreference : OaU8 {
	Discrete,
	Integrated,
	Cpu,
	ByIndex,
};

// PresentationMode controls what graphics surface the engine is built for.
// See Architecture/OaArchitecture.md §10 (renderer/sink split).
//
//   None       — compute-only engine. No graphics queue, no swapchain.
//                Headless training, batch ML inference, CI.
//   Headless   — graphics queue requested, but no VK_KHR_swapchain extension
//                and no surface ever attached. HasGraphics=true, HasPresent
//                =false permanently. Render-farm worker, server-side
//                rendering, batch image-sequence export, CI for graphics
//                code paths. Sinks: SaveImage / EncodeFrame.
//   Swapchain  — graphics queue + swapchain extension. Surface attached via
//                two-phase init (InitPresentation). GUI mode. Sinks:
//                Present (+ optionally SaveImage / EncodeFrame).
enum class OaPresentationMode : OaU8 {
	None,
	Headless,
	Swapchain,
};

// Numeric stability mode.
//
// Fast (default)         vendor math, BF16 CoopMat where supported, DGC
//                        enabled. Maximum throughput.
// Stable                 FP32 path, no CoopMat, no DGC. Used for
//                        training-correctness CI and accuracy
//                        regression where vendor-dependent rounding is a hazard.
// Deterministic          everything in Stable plus: no persistent loop, no
//                        atomics. Used for reproducibility
//                        tests. The slowest path; intentional.
//
// Each mode translates internally to the equivalent env-knob state at engine
// init (OA_FORCE_PRECISION, OA_DISABLE_COOPMAT, OA_DISABLE_DGC,
// OA_DISABLE_PERSISTENT_LOOP). Env knobs set on the command line still override
// the mode-translated defaults — same precedence as OA_DISABLE_COOPMAT overrides
// OaPrecision::BF16 today.
enum class OaNumericMode : OaU8 {
	Fast          = 0,
	Stable        = 1,
	Deterministic = 2,
};

class OaEngineConfig {
public:
	OaDevicePreference  DevicePref       = OaDevicePreference::Discrete;
	OaPresentationMode  PresentationMode = OaPresentationMode::None;
	OaPrecision         Precision        = OaPrecision::FP32;
	OaNumericMode       NumericMode      = OaNumericMode::Fast;
	void*               Surface          = nullptr;  // unused in two-phase flow

#ifdef OA_VULKAN_VALIDATION
	OaBool EnableValidation = true;
#else
	OaBool EnableValidation = false;
#endif
	OaBool   EnablePipelineCache = true;
	// Eagerly instantiate every embedded compute pipeline during engine init.
	// Mobile and other memory-constrained embedders can disable this and rely on
	// the normal on-demand path while preserving the desktop default.
	OaBool   PreloadEmbeddedPipelines = true;
	OaString PipelineCacheDir    = OaPaths::Var("vk").String();
	OaString AppName             = "OaApp";
	OaU32    AppVersion          = 1;
	// SDL3: fill from SDL_Vulkan_GetInstanceExtensions (VK_KHR_surface + platform ext).
	OaVec<OaString> InstanceExtraExtensions;

	OaBool       EnableMultiDevice = false;
	OaU32        DeviceIndex       = 0;
	OaU32        MaxDevices        = 8;
	OaVec<OaU32> MeshVulkanIndices;

	OaBool RegisterAsGlobal = true;

	// Extensions to register during Create (kernels populated into dynamic table).
	// Populated by OaComputeApp::AddExtension or set directly on the config.
	OaVec<OaExtension*> Extensions;
};


enum class OaEngineState : OaU8 {
	Empty,
	Initializing,
	Ready,
	Failed,
	Destroying,
	Destroyed,
};


// The engine is deliberately concrete and pinned. Optional presentation,
// video and collective services compose with it; they do not subclass it.
class OaEngine {
public:
	OaVkDevice         Device;
	OaVma              Allocator;
	OaPipelineRegistry Pipelines;
	OaBindlessHeap     Bindless;

	// Route cache for per-device GEMM variant selection policy.
	// Stores learned profitability data for shape buckets across training.
	OaGemmRouteCache* GemmRouteCache = nullptr;

	OaEngine();
	// Pinned: the engine owns a VkInstance/VkDevice/VMA/queues/mutexes and self-
	// referential pools, so it must never move. Create returns an owning pointer;
	// hold the engine by reference.
	OaEngine(OaEngine&&)            = delete;
	OaEngine(const OaEngine&)       = delete;
	~OaEngine();

	[[nodiscard]] static OaResult<OaUniquePtr<OaEngine>> Create(const OaEngineConfig& InConfig = {});

	// Two-phase construction primitive: initialize a default-constructed (empty)
	// engine in place. Create() is `OaMakeUniquePtr<OaEngine>()` + this. It
	// is public so callers that want to own the engine as a pinned member/reference
	// (default-construct empty, init once) can do so without a build-then-move —
	// e.g. OaComputeApp::RtStorage_. Calling it on an already-initialized engine is
	// a usage error.
	[[nodiscard]] OaStatus InitInPlace(const OaEngineConfig& InConfig);

	// Explicit shutdown boundary. Close drains engine-owned submissions, releases
	// resources, and reports completion failures. Destroy remains as a temporary
	// compatibility facade for callers that cannot yet propagate status.
	[[nodiscard]] OaStatus Close();
	void Destroy();

	[[nodiscard]] bool HasCompute() const noexcept {
		return State_ == OaEngineState::Ready;
	}
	[[nodiscard]] bool HasGraphics() const noexcept {
		return HasCompute()
			and Device.Queues.GraphicsQueueFamily != OaVkEnumerationIndexUnset;
	}
	[[nodiscard]] bool IsReady() const noexcept {
		return State_ == OaEngineState::Ready;
	}
	[[nodiscard]] OaEngineState GetState() const noexcept { return State_; }
	[[nodiscard]] OaContext& GetContext() const noexcept;

	void LogSelectedDevices();

	[[nodiscard]] static OaEngine* GetGlobal();
	[[nodiscard]] OaStatus EnsurePipeline(
		OaStringView InName, OaSpan<const OaU8> InSpirv, const OaPipelineSpec& InSpec);
	[[nodiscard]] OaStatus EnsureAllEmbeddedLiboaPipelines();
	[[nodiscard]] OaComputePipeline& GetPipeline(OaStringView InName);
	[[nodiscard]] OaComputePipeline& GetPipeline(OaKernelId InKernelId);

	void AddShaderSearchPath(const OaString& InPath);
	[[nodiscard]] const OaVec<OaString>& ShaderSearchPaths() const { return ShaderSearchPaths_; }

	OaVkStream* AcquireStream();
	void        ReleaseStream(OaVkStream* InStream);
	// Boolean provenance check for completion-consuming APIs. This intentionally
	// exposes neither the event's device nor its Vulkan semaphore handle.
	[[nodiscard]] bool OwnsEvent(const OaEvent& InEvent) const noexcept;

	[[nodiscard]] OaStatus SubmitToQueue(void* InQueue, void* InSubmitInfo, void* InFence);
	[[nodiscard]] OaStatus SubmitToQueue2(void* InQueue, const VkSubmitInfo2* InSubmitInfo);
	// Submit one engine-owned buffer copy and return its exact completion. Both
	// buffers must be non-aliased VMA allocations created by this primary engine,
	// carry OA's transfer usage, and be owned by its compute queue family. The
	// helper publishes prior primary-compute device writes to the copy read; a
	// producer on another queue is outside this contract. The caller keeps both
	// allocations alive until completion. A GPU consumer on another queue in
	// the same compute family consumes the event as a timeline wait; another
	// family additionally requires graph-planned release/acquire ownership
	// transfer. The engine must outlive event use.
	// A second in-flight copy is rejected.
	[[nodiscard]] OaResult<OaEvent> CopyBufferAsync(
		const OaVkBuffer& InSrc,
		const OaVkBuffer& InDst,
		OaU64 InSize);

	[[nodiscard]] bool       HasAsyncCompute()    const { return Device.Queues.HasAsyncCompute; }
	OaVkStream*              AcquireAsyncStream();
	void                     ReleaseAsyncStream(OaVkStream* InStream);

	[[nodiscard]] OaResult<OaVkBuffer> AllocBuffer(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocBuffer(OaU64 InSize, OaMemoryPlacement InPlacement);
	[[nodiscard]] OaResult<OaVkBuffer> AllocBufferDevice(OaU64 InSize);
	[[nodiscard]] OaResult<OaVkBuffer> AllocBufferBar(OaU64 InSize);
	[[nodiscard]] OaMemoryPlacement DefaultMatrixPlacement() const noexcept {
		return MatrixPlacement_;
	}
	[[nodiscard]] OaStatus UploadBuffer(
		const OaVkBuffer& InDst, OaU64 InDstOffset, const void* InData, OaU64 InSize);
	// Synchronous primary-compute CPU boundary for non-aliased engine-owned VMA
	// storage.
	// Device-local sources stage-copy; mapped sources enqueue a device-write to
	// host-read barrier. Both paths wait, invalidate as needed, then copy bytes.
	[[nodiscard]] OaStatus ReadbackBuffer(
		const OaVkBuffer& InSrc, OaU64 InSrcOffset, void* OutData, OaU64 InSize);
	void                               FreeBuffer(OaVkBuffer& InOutBuffer);

	OaU32 RegisterBuffer(OaVkBuffer& InOutBuffer);
	OaU32 RegisterBufferForOwnedNode(OaVkBuffer& InOutBuffer);
	void  UpdateBufferDescriptor(const OaVkBuffer& InBuffer);
	void  DeregisterBuffer(OaVkBuffer& InOutBuffer);

	[[nodiscard]] bool       HasSAM()                      const { return Device.Info.Hardware.HasSAM; }
	[[nodiscard]] bool       HasCooperativeMatrix()         const { return Device.Info.Software.HasCooperativeMatrix; }
	[[nodiscard]] bool       HasPipelineLibrary()           const { return Device.Info.Software.HasPipelineLibrary; }
	[[nodiscard]] bool       HasDeviceGeneratedCommands()   const { return Device.Info.Software.HasDeviceGeneratedCommands; }
	[[nodiscard]] bool       IsBlackwell()                  const { return Device.Info.Hardware.DeviceId >= 0x2C00 && Device.Info.Hardware.DeviceId < 0x3000; }
	[[nodiscard]] OaU32      SubgroupSize()          const { return Device.Info.Hardware.SubgroupSize; }
	[[nodiscard]] OaU64      DeviceVramBytes()       const { return Device.Info.Hardware.VramBytes; }

	// Cached OaMatmulRegistry::ComputeCapsMask result. The mask depends only
	// on Software/Hardware info populated at device creation, so it never
	// changes after init. Computed on first read by OaGemmRouter::Select et
	// al; subsequent calls are a single relaxed atomic load. Defined in
	// Source/Private/Oa/Runtime/Gemm/Router.cpp.
	[[nodiscard]] OaU64      GemmCapsMask()          const;
	[[nodiscard]] OaStringView DeviceName()          const { return OaStringView(Device.Info.Hardware.DeviceName); }
	[[nodiscard]] OaPrecision  GetPrecision()        const { return Precision_; }
	[[nodiscard]] OaU32        DtypeSpecConstant()   const { return Precision_ == OaPrecision::BF16 ? 1u : 0u; }

	[[nodiscard]] OaU32         DeviceCount()  const;
	[[nodiscard]] OaDeviceNode* GetNode(OaU32 InIndex);
	[[nodiscard]] OaDeviceNode* GetPrimary();
	[[nodiscard]] OaDeviceNode* GetAuxiliary();
	[[nodiscard]] bool          IsMultiDevice() const;
	[[nodiscard]] OaDeviceMesh* GetMesh() { return Mesh_.get(); }

	OaVkStream* AcquireStreamOn(OaU32 InNodeIndex);
	void        ReleaseStreamOn(OaU32 InNodeIndex, OaVkStream* InStream);

	[[nodiscard]] OaStatus SubmitToNodeQueue(
		OaU32 InNodeIndex, void* InQueue, void* InSubmitInfo, void* InFence,
		OaBool InDispatchAlreadyLoadedForNode = false);

	[[nodiscard]] OaStatus PulseAuxiliaryMeshDemoCompute();

	[[nodiscard]] OaResult<OaVkBuffer> AllocBufferOnNode(OaU32 InNodeIndex, OaU64 InSize);
	void                               FreeBufferOnNode(OaVkBuffer& InOutBuffer);

	OaEngine& operator=(OaEngine&&)      = delete;
	OaEngine& operator=(const OaEngine&) = delete;

protected:
	// OaPresenter shares this when GraphicsQueue == ComputeQueue (common on iGPU).
	std::mutex ComputeQueueMutex_;

private:
	friend class OaVkDispatch;
	friend class OaVkImageDispatchTicket;
	friend class OaPresenter;
	friend class OaBorrowedServiceRetirement;
	friend class OaExecutionPlan;
	friend class OaExecutionSession;
	friend class OaComputeGraph;
	friend class OaUploadRing;
	friend class OaGraphicsStreamLease;

	class OaBufferLeaseRegistry;
	class OaRetiredServiceState {
	public:
		using CompleteFn = OaStatus (*)(void*);
		using ReleaseFn = void (*)(void*);

		OaRetiredServiceState() = default;
		OaRetiredServiceState(
			void* InPayload,
			CompleteFn InComplete,
			ReleaseFn InRelease) noexcept
			: Payload_(InPayload)
			, Complete_(InComplete)
			, Release_(InRelease)
		{}
		OaRetiredServiceState(const OaRetiredServiceState&) = delete;
		OaRetiredServiceState& operator=(const OaRetiredServiceState&) = delete;
		OaRetiredServiceState(OaRetiredServiceState&& InOther) noexcept {
			MoveFrom_(OaStdMove(InOther));
		}
		OaRetiredServiceState& operator=(OaRetiredServiceState&& InOther) noexcept {
			if (this != &InOther) {
				ReleasePayload_();
				MoveFrom_(OaStdMove(InOther));
			}
			return *this;
		}
		~OaRetiredServiceState() { ReleasePayload_(); }

		[[nodiscard]] OaStatus Complete() {
			OaStatus status = Payload_ and Complete_
				? Complete_(Payload_)
				: OaStatus::Ok();
			if (status.IsOk()) ReleasePayload_();
			return status;
		}
		void DetachWithoutRelease() noexcept {
			Payload_ = nullptr;
			Complete_ = nullptr;
			Release_ = nullptr;
		}

	private:
		void MoveFrom_(OaRetiredServiceState&& InOther) noexcept {
			Payload_ = InOther.Payload_;
			Complete_ = InOther.Complete_;
			Release_ = InOther.Release_;
			InOther.Payload_ = nullptr;
			InOther.Complete_ = nullptr;
			InOther.Release_ = nullptr;
		}
		void ReleasePayload_() noexcept {
			if (Payload_ and Release_) Release_(Payload_);
			Payload_ = nullptr;
			Complete_ = nullptr;
			Release_ = nullptr;
		}

		void* Payload_ = nullptr;
		CompleteFn Complete_ = nullptr;
		ReleaseFn Release_ = nullptr;
	};

	struct OaRetiredImageDispatch {
		OaVkStream* Stream = nullptr;
		OaVec<OaU32> StorageImageSlots;
		OaVec<OaU32> SampledImageSlots;
		OaVec<OaU32> SamplerSlots;
		OaVec<VkImageView> ImageViews;
	};
	struct OaRetiredContextBatch {
		OaVkStream* Stream = nullptr;
		OaEvent Completion;
		OaVec<OaUniquePtr<OaComputeGraph>> Graphs;
	};
	enum class OaGraphicsStreamSlotState : OaU8 {
		Free,
		Recording,
		Submitted,
		Retired,
		Quarantined,
	};
	struct OaGraphicsStreamSlot {
		OaUniquePtr<OaVkStream> Stream;
		OaGraphicsStreamSlotState State = OaGraphicsStreamSlotState::Free;
		OaU64 Generation = 0;
		OaEvent Completion;
	};

	void SetAsGlobal();
	void ClearGlobal();
	void ReleaseMeshDemoAuxBuffer();
	void RetireImageDispatch(
		OaVkStream* InStream,
		OaVec<OaU32>&& InStorageImageSlots,
		OaVec<OaU32>&& InSampledImageSlots,
		OaVec<OaU32>&& InSamplerSlots,
		OaVec<VkImageView>&& InImageViews);
	void CollectRetiredImageDispatches_();
	void RetireExecutionPlan(OaUniquePtr<OaComputeGraph>&& InGraph);
	void CollectRetiredExecutionPlans_();
	[[nodiscard]] OaStatus CompleteRetiredExecutionPlans_();
	void RetireContextBatch(
		OaVkStream* InStream,
		const OaEvent& InCompletion,
		OaVec<OaUniquePtr<OaComputeGraph>>&& InGraphs);
	void CollectRetiredContextBatches_();
	[[nodiscard]] OaStatus CompleteRetiredContextBatches_();
	[[nodiscard]] OaVkStream* GraphicsStreamForLease_(
		OaU32 InSlot, OaU64 InGeneration) noexcept;
	[[nodiscard]] OaResult<OaEvent> SubmitGraphicsStream_(
		OaU32 InSlot,
		OaU64 InGeneration,
		OaSpan<const OaEvent> InDependencies);
	[[nodiscard]] OaStatus CancelGraphicsStream_(
		OaU32 InSlot, OaU64 InGeneration);
	[[nodiscard]] OaStatus RecycleGraphicsStream_(
		OaU32 InSlot, OaU64 InGeneration, const OaEvent& InCompletion);
	[[nodiscard]] OaStatus AbandonGraphicsStream_(
		OaU32 InSlot, OaU64 InGeneration);
	void CollectRetiredGraphicsStreams_();
	[[nodiscard]] OaStatus CompleteGraphicsStreams_();
	[[nodiscard]] std::mutex* QueueSubmitMutex_(void* InQueue) noexcept;
	void LockQueueSubmit_(void* InQueue);
	void UnlockQueueSubmit_(void* InQueue);
	void RetireUploadRing(OaUniquePtr<OaRetiredUploadRing>&& InRing);
	[[nodiscard]] OaStatus CompleteRetiredUploadRings_();
	void RetirePresenter(OaUniquePtr<OaRetiredPresenter>&& InPresenter);
	[[nodiscard]] OaStatus CompleteRetiredPresenters_();
	void RetireBorrowedService_(
		void* InPayload,
		OaRetiredServiceState::CompleteFn InComplete,
		OaRetiredServiceState::ReleaseFn InRelease);
	[[nodiscard]] OaStatus CompleteRetiredBorrowedServices_();
	void DetachRetiredBorrowedServices_() noexcept;
	[[nodiscard]] OaSharedPtr<OaVkBuffer> AdoptBufferLease_(
		OaVkBuffer&& InBuffer,
		OaSharedPtr<OaVkBuffer> InBacking = {});
	[[nodiscard]] OaStatus InitInPlaceImpl(const OaEngineConfig& InConfig);

	[[nodiscard]] OaStatus EnsurePipelineOnNode(
		OaU32 InNodeIndex, OaStringView InName,
		OaSpan<const OaU8> InSpirv, const OaPipelineSpec& InSpec);

	OaVec<OaString>  ShaderSearchPaths_;
	OaUniquePtr<OaContext> Context_;
	OaEngineState     State_ = OaEngineState::Empty;
	OaSharedPtr<OaBufferLeaseRegistry> BufferLeaseRegistry_;
	OaPrecision      Precision_ = OaPrecision::FP32;
	OaMemoryPlacement MatrixPlacement_ = OaMemoryPlacement::HostUpload;

	// Cap mask cache (see GemmCapsMask() above). Zero sentinel = not yet
	// computed; a real cap mask always sets kCapTiledFp32 so 0 is unambiguous.
	mutable std::atomic<OaU64> GemmCapsMask_{0};

	OaVec<OaUniquePtr<OaVkStream>> StreamPool_;
	OaVec<OaU32>                   FreeStack_;
	OaSpinlock                     StreamPoolLock_;
	OaVec<OaRetiredImageDispatch>  RetiredImageDispatches_;
	std::mutex                     RetiredImageDispatchMutex_;
	OaVec<OaUniquePtr<OaComputeGraph>> RetiredExecutionPlans_;
	OaMutex                            RetiredExecutionPlanMutex_;
	OaVec<OaRetiredContextBatch>       RetiredContextBatches_;
	OaMutex                            RetiredContextBatchMutex_;
	OaVec<OaUniquePtr<OaRetiredUploadRing>> RetiredUploadRings_;
	OaMutex                                 RetiredUploadRingMutex_;
	OaVec<OaUniquePtr<OaRetiredPresenter>> RetiredPresenters_;
	OaMutex                                RetiredPresenterMutex_;
	OaVec<OaRetiredServiceState> RetiredBorrowedServices_;
	OaMutex                       RetiredBorrowedServiceMutex_;

	OaVec<OaUniquePtr<OaVkStream>> AsyncStreamPool_;
	OaVec<OaU32>                   AsyncFreeStack_;
	OaSpinlock                     AsyncStreamPoolLock_;
	OaVec<OaGraphicsStreamSlot>    GraphicsStreamPool_;
	std::mutex                     GraphicsStreamPoolMutex_;

	OaVkStream                TransferStream_;
	// Synchronous public readback still needs a CPU completion boundary, but its
	// Vulkan command resources and mapped staging allocation are engine-owned and
	// reused. This avoids rebuilding a command pool, timeline semaphore and VMA
	// allocation for every scalar/log/checkpoint read.
	OaVkStream                ReadbackStream_;
	OaVkBuffer                ReadbackStaging_;
	OaUniquePtr<OaUploadRing> UploadRing_;
	OaUniquePtr<OaDeviceMesh> Mesh_;

	std::mutex AsyncComputeQueueMutex_;
	std::mutex TransferQueueMutex_;
	std::mutex GraphicsQueueMutex_;
	std::mutex PresentQueueMutex_;
	std::mutex TransferStreamMutex_;
	std::mutex UploadRingMutex_;
	std::mutex ReadbackMutex_;
	std::mutex HostVisibleBufferCacheMutex_;
	OaVec<OaVkBuffer> HostVisibleBufferCache_;
	OaU64 HostVisibleBufferCacheBytes_ = 0;

	OaVkBuffer MeshDemoAuxBuf_{};
	OaU32      MeshDemoAuxNode_       = 0;
	OaBool     MeshDemoAuxScaleReady_ = false;

};


// ─── Optional presentation service ─────────────────────────────────────────
//
// Surface ownership normally stays with the caller. Call Close() or
// DetachPresentation() before destroying the surface. As misuse containment,
// destroying an attached presenter transfers the surface and WSI state to the
// engine; the engine destroys them at Close() and the caller must not do so.
class OaPresenter {
public:
	explicit OaPresenter(OaEngine& InEngine) noexcept : Engine_(InEngine) {}
	OaPresenter(OaPresenter&&)            = delete;
	OaPresenter(const OaPresenter&)       = delete;
	~OaPresenter();

	OaPresenter& operator=(OaPresenter&&)      = delete;
	OaPresenter& operator=(const OaPresenter&) = delete;

	[[nodiscard]] OaEngine& Engine() const noexcept { return Engine_; }
	[[nodiscard]] bool HasGraphics() const noexcept { return Engine_.HasGraphics(); }
	[[nodiscard]] bool HasPresent()  const noexcept { return Swapchain_.PresentReady; }

	// Phase-C: attach surface, build swapchain + renderpass + sync.
	// Safe to call again after DetachPresentation() with a new surface.
	[[nodiscard]] bool InitPresentation(void* InSurface, VkExtent2D InExtent);

	// Tear down swapchain-dependent resources, clear Surface_.
	// Call BEFORE vkDestroySurfaceKHR on the old surface.
	void DetachPresentation();

	// Phase-D: ImGui SDL3 backend init. Compile with -DOA_IMGUI to activate.
	// InNativeWindow — SDL_Window* from OaVkWindow::GetNativeWindowHandle().
	// Without OA_IMGUI: returns true immediately (no-op).
	[[nodiscard]] bool InitImGui(void* InNativeWindow);
	void               ShutdownImGui();

	// Per-frame ImGui — no-ops without OA_IMGUI.
	void BeginImGuiFrame();   // processes SDL events + ImGui::NewFrame
	void EndImGuiFrame();     // ImGui::Render

	// acquire → record (clear + optional ImGui) → submit → present.
	[[nodiscard]] bool DrawFrame();

	// Explicit presentation primitives. OaViewer uses these directly; compute
	// graph recording has no swapchain or WSI state.
	//
	// Acquire result: indices + handles the caller needs to address the
	// per-frame sync slot and the acquired image. Out parameters keep the
	// header free of additional includes.
	struct AcquireResult {
		OaU32       FrameSlot   = 0;   // = the OaSwapchain::FrameIndex used for this acquire
		OaU32       ImageIndex  = 0;
		VkImage     Image       = VK_NULL_HANDLE;
		VkImageView View        = VK_NULL_HANDLE;
		bool        Recreated   = false; // true if swapchain was recreated; frame should be skipped
	};

	// Acquire the next swapchain image. Waits the InFlightFence at the
	// current FrameIndex, calls vkAcquireNextImageKHR signalling
	// ImageAvailSem[FrameIndex], populates OutResult. Returns false on hard
	// error (surface lost, zero-size window).
	[[nodiscard]] bool AcquireSwapchainImage(OaSwapchain& InSwap, AcquireResult& OutResult);

	// Body of the graphics CB that PresentSwapchainImage builds. All members
	// are optional; precedence (Step 3b.5):
	//
	//   DrawImGui = true        → render pass with loadOp=CLEAR + ImGui draw
	//                              → finalLayout PRESENT_SRC. ImGui-only
	//                              renders today; blit+ImGui composite needs
	//                              a render-pass with loadOp=LOAD (Step 3c).
	//   BlitSrcBuffer != nullptr → UNDEFINED → TRANSFER_DST →
	//                              vkCmdCopyBufferToImage → PRESENT_SRC
	//   ClearRgba != nullptr     → UNDEFINED → TRANSFER_DST →
	//                              vkCmdClearColorImage → PRESENT_SRC
	//   none                    → UNDEFINED → PRESENT_SRC (no content;
	//                              chain smoke-test only)
	//
	// Buffer source is assumed to be packed 4-bytes-per-pixel and the same
	// extent as the swapchain image. Format swizzle: today's swapchain is
	// VK_FORMAT_B8G8R8A8_SRGB but OaTexture is RGBA8; channel-swap +
	// linear→sRGB conversion are tracked as known limitations until the
	// staging-image vkCmdBlitImage path lands.
	struct PresentArgs {
		const OaF32* ClearRgba       = nullptr;  // 4 floats in [0,1]
		void*        BlitSrcBuffer   = nullptr;  // VkBuffer
		void*        BlitSrcImage    = nullptr;  // VkImage (precedence over Buffer)
		OaI32        BlitSrcLayout   = 0;        // current VkImageLayout of BlitSrcImage
		OaU32        BlitSrcWidth    = 0;
		OaU32        BlitSrcHeight   = 0;
		bool         DrawImGui       = false;    // record ImGui draw via render pass
		OaFilter     Filter          = OaFilter::Linear;
		void*        WaitTimelineSemaphore = nullptr; // VkSemaphore
		OaU64        WaitTimelineValue = 0;
	};

	// Build a one-off graphics command buffer per the body described above,
	// then submit on the graphics queue waiting on ImageAvailSem[InFrameSlot],
	// signalling RenderDoneSem[InFrameSlot] with InFlightFence[InFrameSlot];
	// call vkQueuePresentKHR waiting on RenderDoneSem[InFrameSlot]. Advances
	// Swapchain.FrameIndex on success.
	[[nodiscard]] bool PresentSwapchainImage(
		OaSwapchain&       InSwap,
		OaU32              InImageIndex,
		OaU32              InFrameSlot,
		const PresentArgs& InArgs);

	// True when GraphicsQueue and ComputeQueue are the same VkQueue (common on laptops).
	[[nodiscard]] bool UsesMergedGraphicsComputeQueue() const;

	// Compatibility spelling for presentation/ImGui callbacks that must surround
	// a raw Vulkan queue operation. It selects the centralized mutex for the
	// exact compute, graphics, or present queue handle, including distinct queues.
	void LockSharedQueueSubmit(void* InQueue);
	void UnlockSharedQueueSubmit(void* InQueue);

	// Explicit swapchain resize (DrawFrame handles OUT_OF_DATE automatically).
	[[nodiscard]] bool RecreateSwapchain(VkExtent2D InNewExtent);
	// WSI completion is not implied by the render-submission timeline. Until a
	// present-id/fence path is enabled, this is the narrow explicit boundary for
	// destroying presentation-owned resources.
	[[nodiscard]] OaStatus WaitPresentationIdle();

	// Unified graphics/compute frame batch recorded on the graphics queue.
	// Graphics queues also support compute, so Render canvas draws and OaUi
	// compute composition can share one command buffer and timeline edge. Flush
	// returns the exact submission event consumed by presentation resources.
	[[nodiscard]] OaStatus BeginGraphicsBatch();
	[[nodiscard]] OaResult<OaEvent> FlushGraphicsBatch(
		const OaVkTimelineSemaphore* InProducerSemaphore = nullptr,
		OaU64 InProducerValue = 0);
	[[nodiscard]] OaStatus SyncGraphicsBatch();
	[[nodiscard]] OaVkStream* ActiveGraphicsBatchStream() const {
		return GraphicsBatchStream_;
	}

	// Call from the window's SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED handler.
	// DrawFrame will recreate the swapchain at the top of the next frame.
	void NotifyPixelSizeChanged(int InWidthPx, int InHeightPx);

	[[nodiscard]] bool         IsPresentationReady() const { return Swapchain_.PresentReady; }
	[[nodiscard]] bool         IsImGuiReady()        const { return ImGuiReady_; }
	[[nodiscard]] VkRenderPass GetRenderPass()        const { return RenderPass_; }
	[[nodiscard]] VkFormat     SwapFormat()            const { return Swapchain_.Format; }
	[[nodiscard]] VkExtent2D   SwapchainExtent()      const { return Swapchain_.Extent; }

	// Direct access to the swapchain (FinalGlue §3.5). Returns a reference to
	// the engine-owned OaSwapchain; zero-state when no surface is attached.
	// Use IsPresentationReady() / Swapchain().IsValid() to gate.
	[[nodiscard]] const OaSwapchain& Swapchain() const noexcept { return Swapchain_; }
	[[nodiscard]] OaSwapchain&       Swapchain()       noexcept { return Swapchain_; }

	[[nodiscard]] OaStatus Close();
	// Compatibility wrapper that logs Close() failures. Prefer Close() where the
	// shutdown result can be propagated.
	void Destroy();

private:
	OaEngine& Engine_;

	[[nodiscard]] bool BuildSwapchainObjects();
	[[nodiscard]] bool BuildRenderPass();
	[[nodiscard]] bool BuildFramebuffers();
	[[nodiscard]] bool BuildCommandPool();
	[[nodiscard]] bool BuildSyncObjects();
	[[nodiscard]] OaStatus PreparePresentFence(OaSwapchain& InSwap, OaU32 InFrameSlot);
	void FinishPresent(OaSwapchain& InSwap, OaU32 InFrameSlot, VkResult InResult);

	void DestroySwapchainObjects();
	void DestroySyncObjects();
	void DestroyCommandPool();
	void ShutdownImGuiResources_();
	void Abandon_() noexcept;
	[[nodiscard]] bool HasOwnedState_() const noexcept;
	static void LockSharedQueueSubmitCallback_(VkQueue InQueue, void* InUser);
	static void UnlockSharedQueueSubmitCallback_(VkQueue InQueue, void* InUser);

	// WSI swapchain state (handle, format, extent, images, views, per-frame
	// sync, dirty-resize signal). Extracted into a standalone type so other
	// render sinks (SaveImage, EncodeFrame) sit at the same level — see
	// Architecture/OaArchitecture.md §10.
	OaSwapchain Swapchain_;

	// Render pass + framebuffers reference Swapchain_.Views directly; they go
	// away in Step 5 when dynamic rendering replaces VkRenderPass.
	VkRenderPass               RenderPass_  = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> Framebuffers_;

	VkCommandPool                CmdPool_ = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> CmdBufs_;

	VkDescriptorPool ImGuiPool_  = VK_NULL_HANDLE;
	bool             ImGuiReady_ = false;

	static constexpr OaU32 kGraphicsBatchRingSize = 4;
	OaUniquePtr<OaVkStream> GraphicsBatchRing_[kGraphicsBatchRingSize];
	OaVkStream* GraphicsBatchStream_ = nullptr;
	OaU32 GraphicsBatchRingIndex_ = 0;
};
