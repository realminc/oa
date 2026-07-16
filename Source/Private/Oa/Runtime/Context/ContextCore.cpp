// OaContext implementation — Unified execution context for all compute operations

#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/Dispatch.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/GpuTimer.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/ComputeKernel.h>
#include <Oa/Runtime/RuntimeGlobal.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoDecoder.h>

#include <atomic>

#include <vulkan/vulkan.h>

// ═════════════════════════════════════════════════════════════════════════════
// Thread-local default context
// ═════════════════════════════════════════════════════════════════════════════

static thread_local OaContext* sDefaultContext = nullptr;

// ═════════════════════════════════════════════════════════════════════════════
// OaContext Implementation
// ═════════════════════════════════════════════════════════════════════════════

OaContext::OaContext(OaComputeEngine* InRuntime)
	: Runtime_(InRuntime)
	, Graph_(nullptr)
	, Executed_(false)
{
	assert(InRuntime and "Runtime cannot be null");
	Graph_ = new OaComputeGraph();
	
	OA_LOG_INFO(OaLogComponent::Core, "OaContext created");
}

OaContext::~OaContext() {
	// Ensure pending work is complete
	if (not Executed_ and Graph_->NodeCount() > 0) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext destroyed with unexecuted operations (count: %u)",
			Graph_->NodeCount());
		(void)Execute();
		(void)Sync();
	}

	// Drain anything still parked in DeferredGraphs_ — Sync() does this
	// during normal teardown, but if the caller never called Sync (eager
	// teardown path) we still need to release the graphs. Assume the GPU
	// is idle by the time the context dies.
	for (auto* graph : DeferredGraphs_) {
		if (Runtime_) {
			graph->Reset(Runtime_->Device);
		}
		delete graph;
	}
	DeferredGraphs_.Clear();
	for (auto* graph : ReusableGraphs_) {
		if (Runtime_) {
			graph->Destroy(Runtime_->Device);
		}
		delete graph;
	}
	ReusableGraphs_.Clear();

	// Destroy the graph's Vulkan objects (command pool, secondary CB) that
	// are now kept alive across Execute() calls for reuse. The old per-call
	// Reset(device) destroyed these every time; ClearNodes() preserves them.
	if (Runtime_) {
		Graph_->Destroy(Runtime_->Device);
	}
	delete Graph_;
	OA_LOG_INFO(OaLogComponent::Core, "OaContext destroyed");
}

OaContext* OaContext::Create(OaEngine* InEngine) {
	assert(InEngine and "OaContext::Create: null engine");
	// Today every concrete engine is a compute engine (OaComputeEngine or
	// a subclass). The downcast is always safe under that invariant; we
	// dynamic_cast so a future engine type that doesn't inherit
	// OaComputeEngine surfaces as a clear failure rather than UB.
	auto* compute = dynamic_cast<OaComputeEngine*>(InEngine);
	assert(compute and "OaContext::Create: engine is not compute-capable "
		"(must derive from OaComputeEngine)");
	return new OaContext(compute);
}

OaEngine& OaContext::Engine() const noexcept {
	assert(Runtime_ and "OaContext::Engine: runtime is null");
	return *Runtime_;  // OaComputeEngine inherits OaEngine
}

OaGraphicsEngine* OaContext::VkGraphics() const noexcept {
#ifdef OA_ANDROID_ML
	return nullptr;
#else
	return dynamic_cast<OaGraphicsEngine*>(Runtime_);
#endif
}

bool OaContext::HasCompute()    const noexcept { return Runtime_ != nullptr and Runtime_->HasCompute(); }
bool OaContext::HasGraphics()   const noexcept { return Runtime_ != nullptr and Runtime_->HasGraphics(); }
bool OaContext::HasPresent()    const noexcept { return Runtime_ != nullptr and Runtime_->HasPresent(); }
bool OaContext::HasMeshShader() const noexcept { return Runtime_ != nullptr and Runtime_->HasMeshShader(); }
bool OaContext::HasRayTrace()   const noexcept { return Runtime_ != nullptr and Runtime_->HasRayTrace(); }
bool OaContext::IsRemote()      const noexcept { return Runtime_ != nullptr and Runtime_->IsRemote(); }

OaStatus OaContext::Record(const OaComputeDispatchDesc& InDesc) {
	if (not Runtime_ or not Graph_) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Record '%.*s': null runtime/graph",
			static_cast<int>(InDesc.Kernel.Size()), InDesc.Kernel.Data());
		return OaStatus::Error(OaStatusCode::Internal,
			"context record: null runtime or graph");
	}
	if (InDesc.Kernel.Empty()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"context record: empty kernel name");
	}
	if (InDesc.Access.Size() != InDesc.Buffers.Size()) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"OaContext::Record '%.*s': access=%zu buffers=%zu",
			static_cast<int>(InDesc.Kernel.Size()), InDesc.Kernel.Data(),
			InDesc.Access.Size(), InDesc.Buffers.Size());
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"context record: buffer access count mismatch");
	}
	if (not InDesc.BufferOwners.Empty()
		and InDesc.BufferOwners.Size() != InDesc.Buffers.Size())
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"context record: buffer owner count mismatch");
	}
	if (InDesc.PushSize > OA_VK_MAX_PUSH_CONSTANT_BYTES
		or (InDesc.PushSize != 0U and InDesc.PushData == nullptr))
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"context record: invalid push payload");
	}
	if (not OaVkBindlessPushFits(
			static_cast<OaU32>(InDesc.Buffers.Size()), InDesc.PushSize))
	{
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"context record: bindless index header plus push payload exceeds limit");
	}

#ifndef NDEBUG
	const OaString kernelName(InDesc.Kernel);
	if (OaComputeKernelUsesDefaultBindlessPipeline(kernelName.c_str())) {
		const OaU32 declared = OaSpvPushConstantBlockSizeByName(kernelName.c_str());
		if (declared != 0U) {
			const OaU32 assembled =
				static_cast<OaU32>(InDesc.Buffers.Size()) * sizeof(OaU32)
				+ InDesc.PushSize;
			if (assembled != declared) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"Bindless push mismatch for '%s': %u buffers * 4 + %u push "
					"tail = %u, shader declares %u bytes",
					kernelName.c_str(), static_cast<OaU32>(InDesc.Buffers.Size()),
					InDesc.PushSize, assembled, declared);
				if (not OaEnvFlag::IsSet("OA_DISABLE_PUSH_CHECK")) {
					return OaStatus::Error(OaStatusCode::InvalidArgument,
						"context record: bindless push contract mismatch");
				}
			}
		}
	}
#endif

	Graph_->Add(InDesc);
	Executed_ = false;
	return OaStatus::Ok();
}

// ═════════════════════════════════════════════════════════════════════════════
// Graphics sink record APIs — Step 3b SKELETON
// ═════════════════════════════════════════════════════════════════════════════
//
// Stubs only — the real implementations need (i) per-queue routing in
// OaComputeGraph + scheduler, (ii) Vulkan timeline-semaphore sync between
// compute and graphics queues, and (iii) ImGui's RenderDrawData call routed
// through an OaContext-managed VkCommandBuffer. Each lands in its own sub-
// commit before these paths are enabled.
//
// Until then these assert at runtime. The build links cleanly because no
// existing code path calls them.

#include <Oa/Ui/Image.h>  // OaTexture (must be complete to return by value)

#include <cstring>  // std::memcpy for bit_cast-equivalent

// ─── RecordAcquire (Step 3b.3) ───────────────────────────────────────────────
//
// Acquires the next swapchain image and returns a non-owning, image-backed
// OaTexture wrapping it. Pending state (swapchain pointer + image index +
// frame slot) is stashed on the context for the paired RecordPresent. The
// real WSI calls (vkWaitForFences + vkAcquireNextImageKHR) happen here
// synchronously — graph-deferred acquire/present land in Step 3b.4 alongside
// per-queue routing.
OaTexture OaContext::RecordAcquire(OaSwapchain& InSwap) {
#ifdef OA_ANDROID_ML
	(void)InSwap;
	return OaTexture{};
#else
	assert(HasPresent() and
		"OaContext::RecordAcquire: engine has no swapchain attached "
		"(InitPresentation() not called, or PresentationMode != Swapchain)");
	assert(PendingPresent_.Swap == nullptr and
		"OaContext::RecordAcquire: previous acquire still pending — "
		"RecordPresent must be called between RecordAcquire pairs");

	auto* gfx = VkGraphics();
	assert(gfx != nullptr and "OaContext::RecordAcquire: graphics engine missing");

	OaGraphicsEngine::AcquireResult acq;
	if (not gfx->AcquireSwapchainImage(InSwap, acq) or acq.Recreated) {
		// Swapchain unavailable or just-recreated — caller should skip this
		// frame. Return an invalid OaTexture; PendingPresent_ stays cleared so
		// RecordPresent has nothing to do.
		return OaTexture{};
	}

	PendingPresent_.Swap        = &InSwap;
	PendingPresent_.ImageHandle = acq.Image;
	PendingPresent_.ImageIndex  = acq.ImageIndex;
	PendingPresent_.FrameSlot   = acq.FrameSlot;
	PendingPresent_.HasClear    = false;
	PendingPresent_.ClearColor  = OaClearColor{};

	OaTexture tex;
	tex.Image  = static_cast<void*>(acq.Image);
	tex.View   = static_cast<void*>(acq.View);
	tex.Format = static_cast<OaI32>(InSwap.Format);
	tex.Layout = static_cast<OaI32>(VK_IMAGE_LAYOUT_UNDEFINED);  // = 0
	tex.Width  = static_cast<OaI32>(InSwap.Extent.width);
	tex.Height = static_cast<OaI32>(InSwap.Extent.height);
	return tex;
#endif
}

// ─── RecordPresent (Step 3b.3 / 3b.4) ────────────────────────────────────────
//
// Step 3b.3 issued the graphics submit synchronously here. Step 3b.4 defers
// it: RecordPresent only marks PresentQueued on PendingPresent_, and Sync()
// flushes it via FlushPendingPresent() after compute work has finished. This
// gives the typical "compute → blit → present" pattern a single
// well-ordered submission: compute writes the source buffer; the graphics CB
// vkCmdCopyBufferToImage's that buffer into the swap image; then present.
void OaContext::RecordPresent(OaSwapchain& InSwap, const OaTexture& InTarget) {
	assert(HasPresent() and "OaContext::RecordPresent: engine has no swapchain attached");
	assert(PendingPresent_.Swap == &InSwap and
		"OaContext::RecordPresent: swapchain doesn't match the pending acquire");
	assert(InTarget.IsImageBacked() and
		"OaContext::RecordPresent: target must be the image-backed OaTexture "
		"returned by RecordAcquire");
	assert(InTarget.Image == PendingPresent_.ImageHandle and "OaContext::RecordPresent: target VkImage doesn't match the acquired image");

	PendingPresent_.PresentQueued = true;
}

void OaContext::RecordPresent(
	OaSwapchain& InSwap,
	const OaTexture& InTarget,
	const OaVkTimelineSemaphore& InWaitSemaphore,
	OaU64 InWaitValue
) {
	RecordPresent(InSwap, InTarget);
	PendingPresent_.WaitTimelineSemaphore = InWaitSemaphore.Semaphore;
	PendingPresent_.WaitTimelineValue = InWaitValue;
}

void OaContext::SubmitPresent() {
	FlushPendingPresent();
}

// ─── FlushPendingPresent (Step 3b.4) ─────────────────────────────────────────
//
// Drains a pending present at the end of Sync(): compute work has just been
// flushed and waited, so the source buffer for any blit is GPU-coherent.
// Assembles PresentArgs (blit takes precedence over clear) and hands off to
// OaGraphicsEngine::PresentSwapchainImage. Resets PendingPresent_ either
// way (success or skip).
void OaContext::FlushPendingPresent() {
#ifdef OA_ANDROID_ML
	PendingPresent_ = PendingPresent{};
	return;
#else
	if (not PendingPresent_.PresentQueued or PendingPresent_.Swap == nullptr) {
		PendingPresent_ = PendingPresent{};
		return;
	}

	auto* gfx = VkGraphics();
	if (gfx == nullptr) {
		PendingPresent_ = PendingPresent{};
		return;
	}

	OaGraphicsEngine::PresentArgs args;
	const OaF32 rgba[4] = {
		PendingPresent_.ClearColor.R,
		PendingPresent_.ClearColor.G,
		PendingPresent_.ClearColor.B,
		PendingPresent_.ClearColor.A,
	};
	if (PendingPresent_.HasImGuiDraw) {
		args.DrawImGui = true;
	} else if (PendingPresent_.HasBlit) {
		args.BlitSrcImage  = PendingPresent_.BlitSrcImage;
		args.BlitSrcLayout = PendingPresent_.BlitSrcLayout;
		args.BlitSrcBuffer = PendingPresent_.BlitSrcBuffer;
		args.BlitSrcWidth  = PendingPresent_.BlitSrcWidth;
		args.BlitSrcHeight = PendingPresent_.BlitSrcHeight;
	} else if (PendingPresent_.HasClear) {
		args.ClearRgba = rgba;
	}
	args.Filter = PendingPresent_.Filter;
	args.WaitTimelineSemaphore = PendingPresent_.WaitTimelineSemaphore;
	args.WaitTimelineValue = PendingPresent_.WaitTimelineValue;

	(void)gfx->PresentSwapchainImage(
		*PendingPresent_.Swap,
		PendingPresent_.ImageIndex,
		PendingPresent_.FrameSlot,
		args);

	PendingPresent_ = PendingPresent{};
#endif
}

// ─── RecordBlit (Step 3b.2) ──────────────────────────────────────────────────
//
// Implementation note: OaTexture is buffer-backed packed RGBA8 today (each
// pixel = 4 bytes = one f32 slot in the bindless heap). For same-extent
// buffer-backed source and destination, we record a "Copy" dispatch node
// using the existing autogenerated f32 Copy shader (one pixel = one f32
// element). That goes through the normal OaContext graph + ExecuteInAsyncBatch
// pipeline — no immediate dispatch, no special graphics-queue path needed.
//
// Out of scope for 3b.2 (lands in 3b.4 with proper graphics-queue routing
// + VkImage support):
//   * Cross-size blits (scaling) — needs a real vkCmdBlitImage path
//   * Cross-format conversions — same
//   * VkImage-backed source/destination — same; today OaTexture is buffer-
//     backed, swapchain images are VkImage, so the compose→swapchain blit in
//     Step 3c needs the VkImage path
//   * Rect-subset blits (SrcRect / DstRect non-empty)
void OaContext::RecordBlit(const OaBlitDesc& InDesc) {
	assert(HasCompute() and "OaContext::RecordBlit: engine has no compute queue");
	assert(InDesc.Src != nullptr and "OaContext::RecordBlit: Src texture is null");
	assert(InDesc.Dst != nullptr and "OaContext::RecordBlit: Dst texture is null");
	assert(InDesc.Src->IsValid() and "OaContext::RecordBlit: Src texture is invalid (no buffer)");
	assert(InDesc.Dst->IsValid() and "OaContext::RecordBlit: Dst texture is invalid (no buffer)");

	// Image-backed destination: the typical compose→swapchain case. The blit
	// is queued onto the pending-present state and emitted inside the
	// present CB by FlushPendingPresent → PresentSwapchainImage.
	//   * Step 3b.4: buffer-backed src → vkCmdCopyBufferToImage (same extent)
	//   * Step 3c.1: VkImage-backed src → vkCmdBlitImage (cross-size + cross-
	//                format; supersedes the buffer-source path)
	if (InDesc.Dst->IsImageBacked()) {
		assert(PendingPresent_.Swap != nullptr and "OaContext::RecordBlit: image-backed dst with no pending acquire");
		assert(InDesc.Dst->Image == PendingPresent_.ImageHandle and	"OaContext::RecordBlit: image-backed dst doesn't match pending acquire");
		assert(InDesc.SrcRect.IsEmpty() and InDesc.DstRect.IsEmpty() and "OaContext::RecordBlit: rect-subset blit to swap image not yet supported");

		PendingPresent_.HasBlit = true;
		PendingPresent_.Filter   = InDesc.Filter;
		if (InDesc.Src->IsImageBacked()) {
			// VkImage source (compose image / offscreen render target).
			// vkCmdBlitImage handles cross-size + cross-format.
			PendingPresent_.BlitSrcImage  = InDesc.Src->Image;
			PendingPresent_.BlitSrcLayout = InDesc.Src->Layout;
			PendingPresent_.BlitSrcBuffer = nullptr;
		} else {
			// Buffer-backed source — vkCmdCopyBufferToImage; same extent only.
			assert(InDesc.Src->Width == InDesc.Dst->Width and
			       InDesc.Src->Height == InDesc.Dst->Height and
				"OaContext::RecordBlit: cross-size buffer→VkImage blit not yet supported");
			PendingPresent_.BlitSrcBuffer = InDesc.Src->DeviceBuf.Buffer;
			PendingPresent_.BlitSrcImage  = nullptr;
		}
		PendingPresent_.BlitSrcWidth  = static_cast<OaU32>(InDesc.Src->Width);
		PendingPresent_.BlitSrcHeight = static_cast<OaU32>(InDesc.Src->Height);
		// Blit supersedes clear within the same frame.
		PendingPresent_.HasClear = false;
		return;
	}

	// Buffer→buffer same-size path (Step 3b.2).
	assert(not InDesc.Src->IsImageBacked() and
		"OaContext::RecordBlit: VkImage source into buffer dst not supported");
	assert(InDesc.Src->Width == InDesc.Dst->Width and InDesc.Src->Height == InDesc.Dst->Height and
		"OaContext::RecordBlit: cross-size blit not yet supported (Step 3b.4)");
	assert(InDesc.SrcRect.IsEmpty() and InDesc.DstRect.IsEmpty() and
		"OaContext::RecordBlit: rect-subset blit not yet supported (Step 3b.4)");

	const OaU32 pixelCount = static_cast<OaU32>(InDesc.Src->Width) * static_cast<OaU32>(InDesc.Src->Height);
	if (pixelCount == 0) return;

	OaVkBuffer bufs[2]   = { InDesc.Src->DeviceBuf, InDesc.Dst->DeviceBuf };
	OaBufferAccess acc[2] = { OaBufferAccess::Read, OaBufferAccess::Write };
	struct { OaU32 Count; } push{ pixelCount };

	// 256-thread workgroup, one element per thread; matches the autogenerated
	// Copy.gen.slang [numthreads(256, 1, 1)].
	constexpr OaU32 kGroupSize = 256;
	const OaU32 groupsX = (pixelCount + kGroupSize - 1) / kGroupSize;

	Add("Copy", OaSpan<OaVkBuffer>(bufs, 2), OaSpan<OaBufferAccess>(acc, 2),
		&push, sizeof(push), groupsX);
}

// ─── RecordClear (Step 3b.2) ─────────────────────────────────────────────────
//
// Implementation: packs OaClearColor (R, G, B, A floats in [0,1]) into a u32
// RGBA8 value (one byte per channel), then writes that bit pattern into every
// pixel slot via the autogenerated "Fill" shader. The Fill kernel takes a
// float Value; reinterpreting the u32 bits as a float is exact for any
// 4-byte payload, so the resulting bytes in memory match the packed color.
//
// Out of scope for 3b.2 (lands in 3b.4):
//   * Non-RGBA8 OaTexture (HDR, planar, depth/stencil)
//   * VkImage-backed targets — needs vkCmdClearColorImage path
//   * Rect-subset clears
void OaContext::RecordClear(const OaTexture& InTarget, OaClearColor InColor) {
	assert(HasCompute() and "OaContext::RecordClear: engine has no compute queue");
	assert(InTarget.IsValid() and "OaContext::RecordClear: target texture is invalid");

	// Image-backed branch (Step 3b.3): when the target is an OaTexture returned
	// by RecordAcquire, the clear is queued into the pending-present state and
	// emitted as vkCmdClearColorImage by RecordPresent (which owns the graphics
	// CB). The buffer-backed Fill compute path below is unchanged.
	if (InTarget.IsImageBacked()) {
		assert(PendingPresent_.Swap != nullptr and
			"OaContext::RecordClear: image-backed target with no pending acquire");
		assert(InTarget.Image == PendingPresent_.ImageHandle and
			"OaContext::RecordClear: image-backed target doesn't match pending acquire");
		PendingPresent_.HasClear   = true;
		PendingPresent_.ClearColor = InColor;
		return;
	}

	const OaU32 pixelCount = static_cast<OaU32>(InTarget.Width) * static_cast<OaU32>(InTarget.Height);
	if (pixelCount == 0) {
		return;
	}

	// Clamp + pack OaClearColor → RGBA8 → u32 (little-endian: R is low byte).
	auto clamp01 = [](OaF32 v) -> OaU32 {
		v = v < 0.0F ? 0.0F : (v > 1.0F ? 1.0F : v);
		return static_cast<OaU32>(v * 255.0F + 0.5F) & 0xFFU;
	};
	const OaU32 packed =
		(clamp01(InColor.A) << 24) |
		(clamp01(InColor.B) << 16) |
		(clamp01(InColor.G) <<  8) |
		(clamp01(InColor.R));
	OaF32 fillValue = 0.0F;
	std::memcpy(&fillValue, &packed, sizeof(fillValue));  // bit-cast

	OaVkBuffer bufs[2]   = { InTarget.DeviceBuf, InTarget.DeviceBuf };  // see Fill convention
	OaBufferAccess acc[2] = { OaBufferAccess::Read, OaBufferAccess::Write };
	struct { OaU32 Count; OaF32 Value; } push{ pixelCount, fillValue };

	constexpr OaU32 kGroupSize = 256;
	const OaU32 groupsX = (pixelCount + kGroupSize - 1) / kGroupSize;

	Add("Fill", OaSpan<OaVkBuffer>(bufs, 2), OaSpan<OaBufferAccess>(acc, 2),
		&push, sizeof(push), groupsX);
}

// ─── RecordImGuiFrame (Step 3b.5) ────────────────────────────────────────────
//
// Deferred: marks PendingPresent_.HasImGuiDraw and lets FlushPendingPresent
// route to PresentArgs.DrawImGui. The actual
// ImGui_ImplVulkan_RenderDrawData() call happens inside PresentSwapchainImage
// (Phase-1 Vulkan-backend hook). The user must have already called
// engine.BeginImGuiFrame() / built their ImGui draws / engine.EndImGuiFrame()
// — the latter calls ImGui::Render() to populate the draw data that
// PresentSwapchainImage reads via ImGui::GetDrawData().
//
// Today the path is ImGui-only (loadOp=CLEAR + ImGui draw). Blit+ImGui
// composite (compute-produced background + ImGui overlay) needs a render
// pass with loadOp=LOAD and lands with Step 3c when DrawFrame's body is
// replaced wholesale by ctx-mediated present.
void OaContext::RecordImGuiFrame(const OaTexture& InTarget) {
	assert(HasGraphics() and "OaContext::RecordImGuiFrame: engine has no graphics queue");
#ifndef OA_IMGUI
	// No-op when ImGui isn't compiled in (Phase 2 will replace this entirely
	// with compute-shader widgets per the bridge doc).
	(void)InTarget;
#else
	assert(PendingPresent_.Swap != nullptr and
		"OaContext::RecordImGuiFrame: no pending acquire — call RecordAcquire first");
	assert(InTarget.IsImageBacked() and
		"OaContext::RecordImGuiFrame: target must be the image-backed OaTexture "
		"returned by RecordAcquire");
	assert(InTarget.Image == PendingPresent_.ImageHandle and
		"OaContext::RecordImGuiFrame: target VkImage doesn't match the acquired image");
	assert(not PendingPresent_.HasBlit and not PendingPresent_.HasClear and
		"OaContext::RecordImGuiFrame: blit+ImGui / clear+ImGui composite not yet "
		"supported (Step 3c)");
	PendingPresent_.HasImGuiDraw = true;
#endif
}

// ─── RecordEncode (FinalGlue Step 3g.4) ──────────────────────────────────────
//
// Phase-1 implementation: synchronous wrapper around the encoder's existing
// UploadInputRgba + EncodeFrame entry points. The OaTexture's RGBA8 bindless
// slot is passed straight to the encoder; the encoder dispatches the
// CvtRgbaToNv12 conversion shader to fill its NV12 input picture, then
// runs the H.264 encode command sequence and packs the bitstream into
// OutFrame. No graph-deferred path yet — that comes when ImageDispatch
// itself records into ctx.

OaStatus OaContext::RecordEncode(
	OaVideoEncoder& InEncoder,
	const OaTexture& InRgba,
	OaU64 InPts,
	OaEncodedFrame& OutFrame
) {
#ifdef OA_ANDROID_ML
	(void)InEncoder;
	(void)InRgba;
	(void)InPts;
	(void)OutFrame;
	return OaStatus::Unimplemented("Video encoding is not part of the Android ML profile");
#else
	if (not InRgba.IsValid()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,	"OaContext::RecordEncode: source OaTexture is invalid");
	}
	// The encoder reads RGBA from the bindless storage-buffer heap, so the
	// source has to be buffer-backed (OaTexture::FromPixels / LoadFile both
	// produce buffer-backed textures — image-backed will land with the
	// VkImage-backed OaTexture type in Step 5+).
	if (InRgba.IsImageBacked()) {
		return OaStatus::Error(OaStatusCode::InvalidArgument,
			"OaContext::RecordEncode: source OaTexture must be buffer-backed "
			"(image-backed sources land with the VkImage OaTexture in Step 5+)");
	}
	OaStatus uploadStatus = InEncoder.UploadInputRgba(
		InRgba.DeviceBuf, InRgba.Width, InRgba.Height,
		OaYCbCrModel::BT709, false);
	if (not uploadStatus.IsOk()) {
		return uploadStatus;
	}
	// Flush any compute work in the current graph so the encoder sees the
	// NV12 input image with all writes visible. Phase 2 routes the dispatch
	// through ctx so this becomes implicit; Phase 1 just submits now.
	OaStatus execStatus = Execute();
	if (not execStatus.IsOk()) { return execStatus; }
	OaStatus syncStatus = Sync();
	if (not syncStatus.IsOk()) { return syncStatus; }
	return InEncoder.EncodeFrame(VK_NULL_HANDLE, InPts, OutFrame);
#endif
}


// ─── RecordDecode (Phase-1 symmetric to RecordEncode) ────────────────────────
//
// Drains any pending compute work, then delegates to the decoder's existing
// DecodeFrameWithConversion entry point. Graph-deferred recording arrives
// when the decoder learns to push its commands through OaContext rather than
// owning its own command buffer.

OaStatus OaContext::RecordDecode(
	OaVideoDecoder& InDecoder,
	const OaSpan<const OaU8>& InAccessUnit,
	const OaVideoConversionOptions& InOptions,
	OaU64 InPts,
	OaVideoFrame& OutFrame
) {
#ifdef OA_ANDROID_ML
	(void)InDecoder;
	(void)InAccessUnit;
	(void)InOptions;
	(void)InPts;
	(void)OutFrame;
	return OaStatus::Unimplemented("Video decoding is not part of the Android ML profile");
#else
	if (not InDecoder.IsInitialized()) {
		return OaStatus::Error("OaContext::RecordDecode: decoder session not initialized");
	}
	if (InDecoder.GetEngine() != &Engine()) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"OaContext::RecordDecode: decoder belongs to a different engine"
		);
	}

	// Decode currently submits outside the compute graph. Only drain the
	// context when there is actual pending work that may produce decoder
	// inputs or share resources. The old unconditional Execute()+Sync()
	// added a full context synchronization to every otherwise-independent
	// video frame.
	if (NodeCount() > 0) {
		OA_RETURN_IF_ERROR(Execute());
	}
	if (IsAsyncBatchActive()) {
		OA_RETURN_IF_ERROR(Sync());
	}

	OaStatus status = InDecoder.DecodeFrameWithConversion(
		InAccessUnit,
		InOptions,
		OutFrame
	);
	if (status.IsOk()) {
		OutFrame.PresentationTimestamp = InPts;
	}
	return status;
#endif
}


void OaContext::SetDefault(OaContext* InContext) {
	sDefaultContext = InContext;
}

OaContext& OaContext::GetDefault() {
	assert(sDefaultContext and "Default context not set. Call OaContext::SetDefault() or initialize engine.");
	return *sDefaultContext;
}

void OaContext::BeginStableResourceFrame() {
	assert(not StableResourceFrameActive_ and
		"stable resource frames cannot be nested");
	StableResourceCursor_ = 0;
	StableResourceFrameActive_ = true;
}

void OaContext::EndStableResourceFrame() {
	StableResourceCursor_ = 0;
	StableResourceFrameActive_ = false;
}

OaSharedPtr<OaVkBuffer> OaContext::AllocateMatrixBuffer(OaU64 InBytes) {
	if (not Runtime_ or InBytes == 0) return {};
	static const OaBool logStableResourceMisses =
		OaEnvFlag::IsSet("OA_LOG_STABLE_RESOURCE_MISSES");

	const auto allocate = [&]() -> OaSharedPtr<OaVkBuffer> {
		auto result = Runtime_->AllocBuffer(InBytes);
		if (not result) return {};
		return OaSharedPtr<OaVkBuffer>(
			new OaVkBuffer(std::move(*result)),
			[](OaVkBuffer* InPtr) {
				if (not InPtr) return;
				if (auto* runtime = OaRuntimeGlobal::GetRuntime()) {
					runtime->FreeBuffer(*InPtr);
				}
				delete InPtr;
			});
	};

	if (not StableResourceFrameActive_) return allocate();

	const OaUsize slot = StableResourceCursor_++;
	if (slot < StableResourceSlots_.Size()) {
		auto& existing = StableResourceSlots_[slot];
		// Stable frames deliberately reuse storage by allocation ordinal. A caller
		// may retain the matrix object itself between frames (for example a batch
		// buffer that is refilled every step); that does not change slot identity.
		if (existing and existing->Size >= InBytes) {
			return existing;
		}
		if (logStableResourceMisses) {
			OA_LOG_INFO(OaLogComponent::Core,
				"Stable resource slot %zu replaced: %llu -> %llu bytes",
				slot,
				static_cast<unsigned long long>(existing ? existing->Size : 0),
				static_cast<unsigned long long>(InBytes));
		}
		existing = allocate();
		return existing;
	}

	auto buffer = allocate();
	if (logStableResourceMisses) {
		OA_LOG_INFO(OaLogComponent::Core,
			"Stable resource slot %zu created: %llu bytes", slot,
			static_cast<unsigned long long>(InBytes));
	}
	StableResourceSlots_.PushBack(buffer);
	return buffer;
}

// ═════════════════════════════════════════════════════════════════════════════
// OaContext dispatch recording
// ═════════════════════════════════════════════════════════════════════════════

void OaContext::Add(
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	OaComputeDispatchDesc desc;
	desc.Kernel = InKernelName;
	desc.Buffers = InBuffers;
	desc.Access = InAccess;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.GroupsX = InGroupsX;
	desc.GroupsY = InGroupsY;
	desc.GroupsZ = InGroupsZ;
	(void)Record(desc);
}

void OaContext::Add(
	OaStringView InKernelName,
	std::initializer_list<const OaMatrix*> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY,
	OaU32 InGroupsZ
) {
	OaVec<OaVkBuffer> buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> owners;
	buffers.Reserve(InMatrices.size());
	owners.Reserve(InMatrices.size());

	OaU32 dtype = 0;
	for (const OaMatrix* matrix : InMatrices) {
		if (matrix) {
			const OaScalarType scalarType = matrix->GetDtype();
			if (scalarType == OaScalarType::BFloat16 or scalarType == OaScalarType::Float16) {
				dtype = 1;
			}
		}
		if (matrix and matrix->VkBuf_) {
			buffers.PushBack(*matrix->VkBuf_);
			owners.PushBack(matrix->VkBuf_);
		} else {
			buffers.PushBack(OaVkBuffer{});
			owners.PushBack({});
		}
	}

	OaComputeDispatchDesc desc;
	desc.Kernel = InKernelName;
	desc.Buffers = buffers.Span();
	desc.BufferOwners = owners.Span();
	desc.Access = InAccess;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.Dtype = dtype;
	desc.GroupsX = InGroupsX;
	desc.GroupsY = InGroupsY;
	desc.GroupsZ = InGroupsZ;
	(void)Record(desc);
}

void OaContext::AddIndirect(
	OaStringView InKernelName,
	std::initializer_list<const OaMatrix*> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	const OaMatrix& InIndirectArgs,
	OaU64 InOffset
) {
	OaVec<OaVkBuffer> buffers;
	OaVec<OaSharedPtr<OaVkBuffer>> owners;
	buffers.Reserve(InMatrices.size());
	owners.Reserve(InMatrices.size());

	OaU32 dtype = 0;
	for (const OaMatrix* matrix : InMatrices) {
		if (matrix) {
			const OaScalarType scalarType = matrix->GetDtype();
			if (scalarType == OaScalarType::BFloat16 or scalarType == OaScalarType::Float16) {
				dtype = 1;
			}
		}
		if (matrix and matrix->VkBuf_) {
			buffers.PushBack(*matrix->VkBuf_);
			owners.PushBack(matrix->VkBuf_);
		} else {
			buffers.PushBack(OaVkBuffer{});
			owners.PushBack({});
		}
	}

	OaComputeDispatchDesc desc;
	desc.Kernel = InKernelName;
	desc.Buffers = buffers.Span();
	desc.BufferOwners = owners.Span();
	desc.Access = InAccess;
	desc.PushData = InPush;
	desc.PushSize = InPushSize;
	desc.Dtype = dtype;
	desc.IndirectBuffer = InIndirectArgs.GetVkBuffer();
	desc.IndirectOffset = InOffset;
	desc.Indirect = true;
	(void)Record(desc);
}
