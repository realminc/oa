// Runtime engine stack (sequential single inheritance):
//   OaEngine            — host-level base (topology/mesh policy hooks; no Vulkan here).
//   OaComputeEngine   — Vulkan compute (device, VMA, bindless, pipelines, queues).
//   OaGraphicsEngine  — same GPU + graphics queue + WSI / swapchain + optional ImGui integration.
//
// Two-phase render init (SDL3 example)
// ─────────────────────────────────────
//   // Phase A — instance + graphics-capable device, no surface yet:
//   OaEngineConfig cfg;
//   cfg.PresentationMode = OaPresentationMode::Swapchain;
//   win.GetPresenterInstanceExtensions(&exts);         // SDL3 fills this
//   for (auto e : exts) cfg.InstanceExtraExtensions.PushBack(e);
//   auto eng = OaGraphicsEngine::Create(cfg).GetValue();
//
//   // Phase B — caller creates surface against the live VkInstance:
//   VkSurfaceKHR surf = VK_NULL_HANDLE;
//   win.CreatePresenterVkSurface(
//       static_cast<VkInstance>(eng.Device.Instance), &surf);
//
//   // Phase C — attach surface, build swapchain:
//   int w, h;  SDL_GetWindowSizeInPixels(sdlWin, &w, &h);
//   eng.InitPresentation(surf, { uint32_t(w), uint32_t(h) });
//
//   // Phase D — optional ImGui (compile with -DOA_IMGUI):
//   eng.InitImGui(win.GetNativeWindowHandle());
//
// Per-frame:
//   eng.BeginImGuiFrame();   // no-op without OA_IMGUI
//   // ImGui / imnodes calls
//   eng.EndImGuiFrame();     // no-op without OA_IMGUI
//   eng.DrawFrame();
//
// Window swap (e.g. splash → editor):
//   VkInstance inst = static_cast<VkInstance>(eng.Device.Instance);
//   eng.DetachPresentation();                           // swapchain torn down
//   vkDestroySurfaceKHR(inst, oldSurf, nullptr);        // caller destroys surface
//   win.reset(); newWin = CreateSdl3(newCfg);
//   newWin.CreatePresenterVkSurface(inst, &newSurf);
//   eng.InitPresentation(newSurf, { uint32_t(w), uint32_t(h) });

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Device.h>
#include <Oa/Core/Extension.h>
#include <Oa/Core/FileIo.h>
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
class OaUploadRing;
class OaContext;


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

// Numeric stability mode — see Docs/Rewrite/Opus/OaNumericStability.md §3.
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
	OaString PipelineCacheDir    = OaFileIo::GetVarDir("vk").String();
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


// A: OaEngine — host / policy base
//
// Capability accessors are virtual so OaContext (and other engine-agnostic
// recorders) can gate work on capabilities without downcasting. Defaults
// here describe a do-nothing base; OaComputeEngine sets HasCompute = true,
// OaGraphicsEngine adds HasGraphics / HasPresent.
class OaEngine {
public:
	virtual ~OaEngine() = default;
	OaEngine(const OaEngine&)            = delete;
	OaEngine& operator=(const OaEngine&) = delete;

	// ─── Capability accessors ────────────────────────────────────────────────
	// Defaults: nothing. Overridden by concrete engine subclasses.
	[[nodiscard]] virtual bool HasCompute()    const noexcept { return false; }
	[[nodiscard]] virtual bool HasGraphics()   const noexcept { return false; }
	[[nodiscard]] virtual bool HasPresent()    const noexcept { return false; }
	[[nodiscard]] virtual bool HasMeshShader() const noexcept { return false; }
	[[nodiscard]] virtual bool HasRayTrace()   const noexcept { return false; }
	[[nodiscard]] virtual bool IsRemote()      const noexcept { return false; }

protected:
	OaEngine() = default;
	OaEngine(OaEngine&&) noexcept            = default;
	OaEngine& operator=(OaEngine&&) noexcept = default;
};

enum class OaEngineState : OaU8 {
	Empty,
	Initializing,
	Ready,
	Failed,
	Destroying,
	Destroyed,
};


// B: OaComputeEngine
class OaComputeEngine : public OaEngine {
public:
	OaVkDevice         Device;
	OaVma              Allocator;
	OaPipelineRegistry Pipelines;
	OaBindlessHeap     Bindless;

	// Route cache for per-device GEMM variant selection policy.
	// Stores learned profitability data for shape buckets across training.
	OaGemmRouteCache* GemmRouteCache = nullptr;

	OaComputeEngine() = default;
	// Pinned: the engine owns a VkInstance/VkDevice/VMA/queues/mutexes and self-
	// referential pools, so it must never move. Create returns an owning pointer;
	// hold the engine by reference.
	OaComputeEngine(OaComputeEngine&&)            = delete;
	OaComputeEngine(const OaComputeEngine&)       = delete;
	~OaComputeEngine() override;

	[[nodiscard]] static OaResult<OaUniquePtr<OaComputeEngine>> Create(const OaEngineConfig& InConfig = {});

	// Two-phase construction primitive: initialize a default-constructed (empty)
	// engine in place. Create() is `OaMakeUniquePtr<OaComputeEngine>()` + this. It
	// is public so callers that want to own the engine as a pinned member/reference
	// (default-construct empty, init once) can do so without a build-then-move —
	// e.g. OaComputeApp::RtStorage_. Calling it on an already-initialized engine is
	// a usage error. OaGraphicsEngine::Create uses it to init its compute base.
	[[nodiscard]] OaStatus InitInPlace(const OaEngineConfig& InConfig);

	virtual void Destroy();

	// ─── Capability overrides ────────────────────────────────────────────────
	[[nodiscard]] bool HasCompute() const noexcept override {
		return State_ == OaEngineState::Ready;
	}
	[[nodiscard]] bool IsReady() const noexcept {
		return State_ == OaEngineState::Ready;
	}
	[[nodiscard]] OaEngineState GetState() const noexcept { return State_; }
	[[nodiscard]] OaContext& GetContext() const noexcept;

	void LogSelectedDevices();

	[[nodiscard]] static OaComputeEngine* GetGlobal();
	// Buffer owners keep only a weak copy of this token. It lets their deleters
	// call back into the exact engine that allocated them while it is alive,
	// without dereferencing a destroyed non-global engine.
	[[nodiscard]] OaWeakPtr<OaU8> GetLifetimeToken() const noexcept {
		return OaWeakPtr<OaU8>(LifetimeToken_);
	}

	[[nodiscard]] OaStatus EnsurePipeline(
		OaStringView InName, OaSpan<const OaU8> InSpirv, const OaPipelineSpec& InSpec);
	[[nodiscard]] OaStatus EnsureAllEmbeddedLiboaPipelines();
	[[nodiscard]] OaComputePipeline& GetPipeline(OaStringView InName);
	[[nodiscard]] OaComputePipeline& GetPipeline(OaKernelId InKernelId);

	void AddShaderSearchPath(const OaString& InPath);
	[[nodiscard]] const OaVec<OaString>& ShaderSearchPaths() const { return ShaderSearchPaths_; }

	OaVkStream* AcquireStream();
	void        ReleaseStream(OaVkStream* InStream);

	[[nodiscard]] OaStatus SubmitToQueue(void* InQueue, void* InSubmitInfo, void* InFence);
	[[nodiscard]] OaStatus SubmitToQueue2(void* InQueue, const VkSubmitInfo2* InSubmitInfo);
	[[nodiscard]] OaStatus CopyBufferAsync(const OaVkBuffer& InSrc, const OaVkBuffer& InDst, OaU64 InSize);
	[[nodiscard]] OaStatus WaitTransfer();

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
	[[nodiscard]] OaStatus ReadbackBuffer(
		const OaVkBuffer& InSrc, OaU64 InSrcOffset, void* OutData, OaU64 InSize);
	void                               FreeBuffer(OaVkBuffer& InOutBuffer);

	OaU32 RegisterBuffer(OaVkBuffer& InOutBuffer);
	OaU32 RegisterBufferForOwnedNode(OaVkBuffer& InOutBuffer);
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

	[[nodiscard]] OaStatus BeginComputeBatch();
	// Non-blocking submit: GPU begins executing while CPU continues.
	// Caller must call SyncCurrentBatch() before reading GPU-written memory.
	[[nodiscard]] OaStatus FlushComputeBatch();
	[[nodiscard]] OaCompletionToken LastComputeBatchCompletion() const;
	// Wait for the most recently submitted batch stream to complete on GPU.
	[[nodiscard]] OaStatus SyncCurrentBatch();
	[[nodiscard]] bool        IsComputeBatchActive()     const { return ComputeBatchStream_ != nullptr; }
	[[nodiscard]] OaVkStream* ActiveComputeBatchStream()  const { return ComputeBatchStream_; }
	[[nodiscard]] OaU32       ComputeBatchRingSize()     const { return kBatchRingSize; }

	[[nodiscard]] OaResult<OaVkBuffer> AllocBufferOnNode(OaU32 InNodeIndex, OaU64 InSize);
	void                               FreeBufferOnNode(OaVkBuffer& InOutBuffer);

	OaComputeEngine& operator=(OaComputeEngine&&)      = delete;
	OaComputeEngine& operator=(const OaComputeEngine&) = delete;

protected:
	// OaGraphicsEngine shares this when GraphicsQueue == ComputeQueue (common on iGPU).
	std::mutex ComputeQueueMutex_;

private:
	friend class OaVkDispatch;

	void SetAsGlobal();
	void ClearGlobal();
	void ReleaseMeshDemoAuxBuffer();
	[[nodiscard]] OaStatus InitInPlaceImpl(const OaEngineConfig& InConfig);

	[[nodiscard]] OaStatus EnsurePipelineOnNode(
		OaU32 InNodeIndex, OaStringView InName,
		OaSpan<const OaU8> InSpirv, const OaPipelineSpec& InSpec);

	OaVec<OaString>  ShaderSearchPaths_;
	OaUniquePtr<OaContext> Context_;
	OaEngineState     State_ = OaEngineState::Empty;
	OaSharedPtr<OaU8> LifetimeToken_ = OaMakeSharedPtr<OaU8>(0);
	OaPrecision      Precision_ = OaPrecision::FP32;
	OaMemoryPlacement MatrixPlacement_ = OaMemoryPlacement::HostUpload;

	// Cap mask cache (see GemmCapsMask() above). Zero sentinel = not yet
	// computed; a real cap mask always sets kCapTiledFp32 so 0 is unambiguous.
	mutable std::atomic<OaU64> GemmCapsMask_{0};

	OaVec<OaUniquePtr<OaVkStream>> StreamPool_;
	OaVec<OaU32>                   FreeStack_;
	OaSpinlock                     StreamPoolLock_;

	OaVec<OaUniquePtr<OaVkStream>> AsyncStreamPool_;
	OaVec<OaU32>                   AsyncFreeStack_;
	OaSpinlock                     AsyncStreamPoolLock_;

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
	std::mutex UploadRingMutex_;
	std::mutex ReadbackMutex_;
	std::mutex HostVisibleBufferCacheMutex_;
	OaVec<OaVkBuffer> HostVisibleBufferCache_;
	OaU64 HostVisibleBufferCacheBytes_ = 0;

	OaVkBuffer MeshDemoAuxBuf_{};
	OaU32      MeshDemoAuxNode_       = 0;
	OaBool     MeshDemoAuxScaleReady_ = false;

	OaVkStream* ComputeBatchStream_ = nullptr;

	// Dedicated batch streams. A deeper ring lets high-level training submit
	// short macro-batches before the CPU has to recycle a command buffer slot.
	static constexpr OaU32 kBatchRingSize = 16;
	OaVkStream* BatchRing_[kBatchRingSize] = {};
	OaU32       BatchRingIdx_              = 0;
};


// ─── C: OaGraphicsEngine ─────────────────────────────────────────────────────
//
// Surface ownership: VkSurfaceKHR is owned by the CALLER, not this engine.
// Call DetachPresentation() before destroying the surface.
class OaGraphicsEngine : public OaComputeEngine {
public:
	OaGraphicsEngine() = default;
	OaGraphicsEngine(OaGraphicsEngine&&)            = delete;   // pinned — see OaComputeEngine
	OaGraphicsEngine(const OaGraphicsEngine&)       = delete;
	~OaGraphicsEngine() override;

	OaGraphicsEngine& operator=(OaGraphicsEngine&&)      = delete;
	OaGraphicsEngine& operator=(const OaGraphicsEngine&) = delete;

	// Phase-A: create device with graphics queue + VK_KHR_swapchain, no surface.
	// cfg.PresentationMode must be Swapchain.
	// cfg.InstanceExtraExtensions must contain SDL3 WSI extension names.
	[[nodiscard]] static OaResult<OaUniquePtr<OaGraphicsEngine>> Create(const OaEngineConfig& InConfig = {});

	// ─── Capability overrides ────────────────────────────────────────────────
	// HasCompute is inherited (true). HasGraphics is true by construction —
	// this engine type owns a graphics queue. HasPresent follows the surface
	// attachment state (false until InitPresentation succeeds; reset on
	// DetachPresentation).
	[[nodiscard]] bool HasGraphics() const noexcept override { return IsReady(); }
	[[nodiscard]] bool HasPresent()  const noexcept override { return Swapchain_.PresentReady; }

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

	// ─── Compatibility context-mediated present primitives ─────────────────────────────────────
	//
	// These split DrawFrame's body into the two halves that OaContext::
	// RecordAcquire and OaContext::RecordPresent call. DrawFrame() above is
	// preserved as a self-driven loop for callers that don't go through ctx
	// (e.g. compact tools that want one-call render-and-present). OaDeviceUi
	// (Step 3c.2) drives presentation through ctx via these primitives.
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
	// error (surface lost, zero-size window). Used by OaContext::RecordAcquire.
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
	// Swapchain.FrameIndex on success. Used by OaContext::RecordPresent.
	[[nodiscard]] bool PresentSwapchainImage(
		OaSwapchain&       InSwap,
		OaU32              InImageIndex,
		OaU32              InFrameSlot,
		const PresentArgs& InArgs);

	// True when GraphicsQueue and ComputeQueue are the same VkQueue (common on laptops).
	[[nodiscard]] bool UsesMergedGraphicsComputeQueue() const;

	// Serialize vkQueueSubmit with OaComputeEngine::SubmitToQueue on that shared queue.
	// Call around raw vkQueueSubmit (Dear ImGui texture path, splash upload, etc.).
	void LockSharedQueueSubmit(void* InQueue);
	void UnlockSharedQueueSubmit(void* InQueue);

	// Explicit swapchain resize (DrawFrame handles OUT_OF_DATE automatically).
	[[nodiscard]] bool RecreateSwapchain(VkExtent2D InNewExtent);

	// Unified graphics/compute frame batch recorded on the graphics queue.
	// Graphics queues also support compute, so Render canvas draws and OaUi
	// compute composition can share one command buffer and timeline edge.
	[[nodiscard]] OaStatus BeginGraphicsBatch();
	[[nodiscard]] OaStatus FlushGraphicsBatch(
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

	void Destroy() override;

private:
	[[nodiscard]] bool BuildSwapchainObjects();
	[[nodiscard]] bool BuildRenderPass();
	[[nodiscard]] bool BuildFramebuffers();
	[[nodiscard]] bool BuildCommandPool();
	[[nodiscard]] bool BuildSyncObjects();

	void DestroySwapchainObjects();
	void DestroySyncObjects();
	void DestroyCommandPool();

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
	OaVkStream GraphicsBatchRing_[kGraphicsBatchRingSize];
	OaBool GraphicsBatchRingValid_[kGraphicsBatchRingSize] = {};
	OaVkStream* GraphicsBatchStream_ = nullptr;
	OaU32 GraphicsBatchRingIndex_ = 0;
};
