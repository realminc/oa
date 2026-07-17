// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Text.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Vision/Detection.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ─── BlitRgba push constants (must match BlitRgba.slang) ─────────────────────
struct BlitRgbaPc {
	OaU32 src_idx;
	OaU32 dst_idx;
	OaU32 src_w;
	OaU32 src_h;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
};

// ─── BlitPlanar push constants (must match BlitPlanar.slang) ─────────────────
struct BlitPlanarPc {
	OaU32 r_idx;
	OaU32 g_idx;
	OaU32 b_idx;
	OaU32 a_idx;
	OaU32 dst_idx;
	OaU32 src_w;
	OaU32 src_h;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dtypes;
	OaU32 channels;
	OaU32 dst_w;
	OaU32 dst_h;
};

struct BlitImageRgbaPc {
	OaU32 src_idx;
	OaU32 dst_idx;
	OaU32 src_w;
	OaU32 src_h;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
};

// ─── Filled/outlined rectangle push constants ───────────────────────────────
struct DrawRectPc {
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaU32 rgba;
};

// Must match DrawRectOutline.slang.
struct DrawRectOutlinePc {
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaU32 thickness;
	OaU32 rgba;
};

struct DrawLinePc {
	OaU32 dst_idx;
	OaF32 x0;
	OaF32 y0;
	OaF32 x1;
	OaF32 y1;
	OaF32 thickness;
	OaU32 rgba;
	OaU32 bounds_x;
	OaU32 bounds_y;
	OaU32 bounds_w;
	OaU32 bounds_h;
};

struct DrawWaveformPc {
	OaU32 envelope_idx;
	OaU32 bins;
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaF32 fraction;
	OaU32 played_rgba;
	OaU32 remaining_rgba;
};

struct DrawPlotLinePc {
	OaU32 values_idx;
	OaU32 count;
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaF32 y_min;
	OaF32 y_max;
	OaU32 rgba;
	OaU32 flags;
};

struct DrawHeatmapPc {
	OaU32 values_idx;
	OaU32 rows;
	OaU32 cols;
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaF32 v_min;
	OaF32 v_max;
	OaU32 colormap;
	OaU32 value_type;
	OaU32 offset_elements;
	OaU32 flags;
};

// ─── DrawRectOutlines push constants (must match DrawRectOutlines.slang) ─────
struct DrawRectOutlinesPc {
	OaU32 rect_idx;
	OaU32 count;
	OaU32 dst_idx;
	OaI32 dst_x;
	OaI32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaI32 clip_x;
	OaI32 clip_y;
	OaU32 clip_w;
	OaU32 clip_h;
	OaU32 thickness;
	OaU32 rgba;
};

// ─── DrawGlyphs push constants (must match DrawGlyphs.slang) ────────────────
struct DrawGlyphsPc {
	OaU32 glyph_idx;
	OaU32 atlas_idx;
	OaU32 count;
	OaU32 dst_idx;
	OaI32 dst_x;
	OaI32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaI32 clip_x;
	OaI32 clip_y;
	OaU32 clip_w;
	OaU32 clip_h;
	OaU32 atlas_w;
	OaU32 atlas_h;
	OaF32 px_range;
};


// ─── Deferred blit command ────────────────────────────────────────────────────

enum class BlitKind : OaU8 {
	Rgba,
	Planar,
	ImageRgba,
	Rect,
	RectOutline,
	Line,
	RectOutlines,
	Glyphs,
	Waveform,
	PlotLine,
	Heatmap,
};

struct BlitCmd {
	BlitKind Kind;
	VkImage SrcImage = VK_NULL_HANDLE;
	VkImageView SrcImageView = VK_NULL_HANDLE;
	VkImageLayout SrcImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	union {
		BlitRgbaPc      Rgba;
		BlitPlanarPc    Planar;
		BlitImageRgbaPc ImageRgba;
		DrawRectPc Rect;
		DrawRectOutlinePc RectOutline;
		DrawLinePc Line;
		DrawRectOutlinesPc RectOutlines;
		DrawGlyphsPc Glyphs;
		DrawWaveformPc Waveform;
		DrawPlotLinePc PlotLine;
		DrawHeatmapPc Heatmap;
	};
};


// ─── OaUi::Impl ────────────────────────────────────────────────────────────────

struct OaUi::Impl {
	OaComputeEngine* Rt = nullptr;

	// Blit pipelines (created in InitBlit).
	OaComputePipeline BlitRgba;
	OaComputePipeline BlitPlanar;
	OaComputePipeline BlitImageRgba;
	OaComputePipeline DrawRect;
	OaComputePipeline DrawRectOutline;
	OaComputePipeline DrawLine;
	OaComputePipeline DrawRectOutlines;
	OaComputePipeline DrawGlyphs;
	OaComputePipeline DrawWaveform;
	OaComputePipeline DrawPlotLine;
	OaComputePipeline DrawHeatmap;

	static constexpr OaU32 kPlotSlotCount = 4;
	static constexpr OaU32 kPlotCapacity = 4096;
	struct PlotSlot {
		OaVkBuffer Buffer;
		OaCompletionToken Completion;
		OaU32 Count = 0;
	};
	struct PlotCache {
		OaU32 Id = 0;
		std::array<PlotSlot, kPlotSlotCount> Slots;
		OaU32 LastSlot = 0;
		OaU32 NextSlot = 0;
	};
	struct UsedPlotSlot {
		OaU32 Cache = 0;
		OaU32 Slot = 0;
	};
	OaVec<PlotCache> Plots;
	OaVec<UsedPlotSlot> UsedPlots;

	PlotSlot* UploadPlotValues(OaU32 InId, const OaF32* InData,
		OaU32 InCount, OaU32& OutCacheIndex, OaU32& OutSlotIndex) {
		if (Rt == nullptr || InData == nullptr || InCount == 0U) return nullptr;
		OutCacheIndex = 0U;
		for (; OutCacheIndex < Plots.Size(); ++OutCacheIndex) {
			if (Plots[OutCacheIndex].Id == InId) break;
		}
		if (OutCacheIndex == Plots.Size()) {
			PlotCache cache;
			cache.Id = InId;
			const auto releaseCache = [&] {
				for (auto& slot : cache.Slots) {
					if (slot.Buffer.Buffer == nullptr) continue;
					Rt->DeregisterBuffer(slot.Buffer);
					Rt->Allocator.Free(slot.Buffer);
				}
			};
			const OaU64 bytes = static_cast<OaU64>(kPlotCapacity)
				* sizeof(OaF32);
			for (auto& slot : cache.Slots) {
				auto buffer = Rt->Allocator.AllocHostVisible(bytes);
				if (!buffer.IsOk()) {
					releaseCache();
					return nullptr;
				}
				slot.Buffer = OaStdMove(*buffer);
				if (Rt->RegisterBuffer(slot.Buffer) == OA_BINDLESS_INVALID) {
					Rt->Allocator.Free(slot.Buffer);
					releaseCache();
					return nullptr;
				}
			}
			Plots.PushBack(OaStdMove(cache));
		}

		auto& cache = Plots[OutCacheIndex];
		OutSlotIndex = kPlotSlotCount;
		for (OaU32 offset = 0; offset < kPlotSlotCount; ++offset) {
			const OaU32 slotIndex = (cache.NextSlot + offset) % kPlotSlotCount;
			const auto& completion = cache.Slots[slotIndex].Completion;
			if (!completion.IsValid() || completion.IsComplete()) {
				OutSlotIndex = slotIndex;
				break;
			}
		}
		if (OutSlotIndex < kPlotSlotCount) {
			auto& slot = cache.Slots[OutSlotIndex];
			slot.Count = std::min(InCount, kPlotCapacity);
			std::memcpy(slot.Buffer.MappedPtr, InData,
				static_cast<OaUsize>(slot.Count) * sizeof(OaF32));
			if (!Rt->Allocator.FlushHostBuffer(slot.Buffer, 0,
				static_cast<OaU64>(slot.Count) * sizeof(OaF32))) return nullptr;
			cache.LastSlot = OutSlotIndex;
			cache.NextSlot = (OutSlotIndex + 1U) % kPlotSlotCount;
		} else if (cache.Slots[cache.LastSlot].Count == 0U) {
			return nullptr;
		}
		OutSlotIndex = cache.LastSlot;
		return &cache.Slots[cache.LastSlot];
	}

	struct SampledImageSlot {
		VkImageView View = VK_NULL_HANDLE;
		VkImageLayout Layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		OaU32 Slot = OA_BINDLESS_INVALID;
	};
	OaVec<SampledImageSlot> SampledImageSlots;

	// Deferred blit list, cleared each frame.
	OaVec<BlitCmd> Blits;

	// Panel stack for layout cursor.
	struct PanelState {
		OaPixelRect Rect;
		OaF32       Cursor;  // current Y offset within panel
	};
	OaVec<PanelState> PanelStack;

	// Cached style stack.
	static constexpr OaU32 kStyleDepth = 32;
	OaUiStyle StyleStack[kStyleDepth];
	OaU32    StyleDepth = 0;
	OaUiStyle DefaultStyle;
};


// ─── OaUi move/dtor ───────────────────────────────────────────────────────────

OaUi::OaUi(OaUi&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_))
	, Input_(InOther.Input_)
{}

OaUi& OaUi::operator=(OaUi&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_  = OaStdMove(InOther.Impl_);
		Input_ = InOther.Input_;
	}
	return *this;
}

OaUi::~OaUi() { Destroy(); }


// ─── Init / Destroy ───────────────────────────────────────────────────────────

OaStatus OaUi::Init(OaComputeEngine& InRt, const OaUiStyle& InStyle) {
	Impl_ = OaStdMakeUnique<Impl>();
	Impl_->Rt = &InRt;
	Impl_->DefaultStyle = InStyle;
	return OaStatus::Ok();
}

void OaUi::Destroy() {
	if (!Impl_) return;
	if (Impl_->Rt) {
		for (auto& plot : Impl_->Plots) {
			for (auto& slot : plot.Slots) {
				if (slot.Buffer.Buffer != nullptr) {
					Impl_->Rt->DeregisterBuffer(slot.Buffer);
					Impl_->Rt->Allocator.Free(slot.Buffer);
				}
			}
		}
		Impl_->Plots.Clear();
		Impl_->UsedPlots.Clear();
		for (const auto& slot : Impl_->SampledImageSlots) {
			if (slot.Slot != OA_BINDLESS_INVALID) {
				Impl_->Rt->Bindless.DeregisterSampledImage(slot.Slot);
			}
		}
		Impl_->SampledImageSlots.Clear();
		Impl_->BlitRgba.Destroy(Impl_->Rt->Device);
		Impl_->BlitPlanar.Destroy(Impl_->Rt->Device);
		Impl_->BlitImageRgba.Destroy(Impl_->Rt->Device);
		Impl_->DrawRect.Destroy(Impl_->Rt->Device);
		Impl_->DrawRectOutline.Destroy(Impl_->Rt->Device);
		Impl_->DrawLine.Destroy(Impl_->Rt->Device);
		Impl_->DrawRectOutlines.Destroy(Impl_->Rt->Device);
		Impl_->DrawGlyphs.Destroy(Impl_->Rt->Device);
		Impl_->DrawWaveform.Destroy(Impl_->Rt->Device);
		Impl_->DrawPlotLine.Destroy(Impl_->Rt->Device);
		Impl_->DrawHeatmap.Destroy(Impl_->Rt->Device);
	}
	Impl_.Reset();
}


// ─── InitBlit ─────────────────────────────────────────────────────────────────
// Creates UI blit pipelines against the engine's unified bindless layout.

OaStatus OaUi::InitBlit(void* /*InComposeImageView*/) {
	if (!Impl_ || !Impl_->Rt) return OaStatus::Error("OaUi: not initialized");
	if (!Impl_->Rt->Bindless.PipelineLayout) return OaStatus::Error("OaUi: bindless layout not initialized");

	OaPipelineSpec spec;
	spec.PushConstantBytes = 128;

	auto createBlitPipeline = [&](const char* InName, OaComputePipeline& OutPipeline) -> OaStatus {
		const OaSpvEntry* spv = OaShaderProviderFind(InName);
		if (!spv) return OaStatus::Error(OaStatusCode::NotFound, "OaUi: blit SPIR-V not found");
		auto res = OaComputePipeline::Create(
			Impl_->Rt->Device,
			OaSpan<const OaU8>(spv->Data, spv->Size),
			spec,
			nullptr,
			Impl_->Rt->Bindless.PipelineLayout);
		if (!res.IsOk()) return res.GetStatus();
		OutPipeline = OaStdMove(*res);
		return OaStatus::Ok();
	};

	OA_RETURN_IF_ERROR(createBlitPipeline("BlitRgba", Impl_->BlitRgba));
	OA_RETURN_IF_ERROR(createBlitPipeline("BlitPlanar", Impl_->BlitPlanar));
	OA_RETURN_IF_ERROR(createBlitPipeline("BlitImageRgba", Impl_->BlitImageRgba));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRect", Impl_->DrawRect));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutline", Impl_->DrawRectOutline));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawLine", Impl_->DrawLine));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutlines", Impl_->DrawRectOutlines));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawGlyphs", Impl_->DrawGlyphs));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawWaveform", Impl_->DrawWaveform));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawPlotLine", Impl_->DrawPlotLine));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawHeatmap", Impl_->DrawHeatmap));
	return OaStatus::Ok();
}


// ─── UpdateBlitImage ──────────────────────────────────────────────────────────

void OaUi::UpdateBlitImage(void* /*InComposeImageView*/) {
	// Compose images are registered directly in OaDeviceUi's bindless
	// storage-image heap (see OaDeviceUi::BuildComposeImage).
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

void OaUi::BeginFrame(OaF32 /*InDeltaMs*/) {
	if (!Impl_) return;
	Impl_->Blits.Clear();
	Impl_->UsedPlots.Clear();
	Impl_->PanelStack.Clear();
	Impl_->StyleDepth = 0;
	Input_.MouseDX = 0.0F;
	Input_.MouseDY = 0.0F;
	Input_.ScrollX = 0.0F;
	Input_.ScrollY = 0.0F;
	Input_.LPressed = false;
	Input_.LReleased = false;
	Input_.HoverId = 0U;
}

bool OaUi::RouteEvent(const OaUiEvent& InEvent) {
	Input_.Modifiers = InEvent.Modifiers;
	switch (InEvent.Type) {
		case OuiEventType::MouseMove:
			Input_.MouseX = InEvent.MouseX;
			Input_.MouseY = InEvent.MouseY;
			Input_.MouseDX += InEvent.MouseDX;
			Input_.MouseDY += InEvent.MouseDY;
			break;
		case OuiEventType::MouseDown:
			Input_.MouseX = InEvent.MouseX;
			Input_.MouseY = InEvent.MouseY;
			if (InEvent.Button == 1) {
				Input_.LButton = true;
				Input_.LPressed = true;
			} else if (InEvent.Button == 2) {
				Input_.MButton = true;
			} else if (InEvent.Button == 3) {
				Input_.RButton = true;
			}
			break;
		case OuiEventType::MouseUp:
			Input_.MouseX = InEvent.MouseX;
			Input_.MouseY = InEvent.MouseY;
			if (InEvent.Button == 1) {
				Input_.LButton = false;
				Input_.LReleased = true;
			} else if (InEvent.Button == 2) {
				Input_.MButton = false;
			} else if (InEvent.Button == 3) {
				Input_.RButton = false;
			}
			break;
		case OuiEventType::MouseScroll:
			Input_.MouseX = InEvent.MouseX;
			Input_.MouseY = InEvent.MouseY;
			Input_.ScrollX += InEvent.ScrollX;
			Input_.ScrollY += InEvent.ScrollY;
			break;
		case OuiEventType::WindowBlur:
			Input_.LButton = false;
			Input_.MButton = false;
			Input_.RButton = false;
			Input_.LReleased = true;
			Input_.ActiveId = 0U;
			break;
		default:
			break;
	}
	return Input_.ActiveId != 0U;
}

void OaUi::RecordRender(VkCommandBuffer InCmd, OaU32 InDstBindlessIdx) {
	if (!Impl_ || Impl_->Blits.Empty()) return;
	if (!Impl_->BlitRgba.Pipeline && !Impl_->BlitPlanar.Pipeline
		&& !Impl_->BlitImageRgba.Pipeline && !Impl_->DrawRect.Pipeline
		&& !Impl_->DrawRectOutline.Pipeline && !Impl_->DrawLine.Pipeline
		&& !Impl_->DrawRectOutlines.Pipeline && !Impl_->DrawGlyphs.Pipeline
		&& !Impl_->DrawWaveform.Pipeline && !Impl_->DrawPlotLine.Pipeline
		&& !Impl_->DrawHeatmap.Pipeline) {
		return;
	}

	VkCommandBuffer cmd = InCmd;
	VkPipelineLayout layout = static_cast<VkPipelineLayout>(Impl_->Rt->Bindless.PipelineLayout);

	auto getSampledImageSlot = [this](VkImageView InView, VkImageLayout InLayout) -> OaU32 {
		for (auto& slot : Impl_->SampledImageSlots) {
			if (slot.View == InView) {
				if (slot.Layout != InLayout) {
					Impl_->Rt->Bindless.UpdateSampledImage(Impl_->Rt->Device, slot.Slot, InView, InLayout);
					slot.Layout = InLayout;
				}
				return slot.Slot;
			}
		}
		OaU32 idx = Impl_->Rt->Bindless.RegisterSampledImage(Impl_->Rt->Device, InView, InLayout);
		if (idx != OA_BINDLESS_INVALID) {
			Impl::SampledImageSlot slot;
			slot.View = InView;
			slot.Layout = InLayout;
			slot.Slot = idx;
			Impl_->SampledImageSlots.PushBack(slot);
		}
		return idx;
	};

	for (BlitCmd& bc : Impl_->Blits) {
		if (bc.Kind == BlitKind::ImageRgba && bc.SrcImageView) {
			bc.ImageRgba.src_idx = getSampledImageSlot(bc.SrcImageView, bc.SrcImageLayout);
		}
	}

	// Images supplied by external producers (video conversion, inference,
	// uploads) may have been written by an earlier compute submission. Queue
	// order alone does not make shader writes visible to this sampled-image
	// read; establish the explicit Vulkan memory dependency before blitting.
	for (const BlitCmd& bc : Impl_->Blits) {
		if (bc.Kind != BlitKind::ImageRgba || bc.SrcImage == VK_NULL_HANDLE) continue;
		VkImageMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
		barrier.oldLayout = bc.SrcImageLayout;
		barrier.newLayout = bc.SrcImageLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = bc.SrcImage;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.imageMemoryBarrierCount = 1;
		dependency.pImageMemoryBarriers = &barrier;
		vkCmdPipelineBarrier2(cmd, &dependency);
	}
	// A heatmap may consume an evaluation matrix written by an earlier compute
	// submission. Establish shader-write -> shader-read visibility without a
	// host wait; queue order supplies execution order, this supplies memory order.
	for (const BlitCmd& bc : Impl_->Blits) {
		if (bc.Kind != BlitKind::Heatmap) continue;
		VkMemoryBarrier2 barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
		barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
		barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
		barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
		VkDependencyInfo dependency{};
		dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers = &barrier;
		vkCmdPipelineBarrier2(cmd, &dependency);
		break;
	}

	VkDescriptorSet set = static_cast<VkDescriptorSet>(Impl_->Rt->Bindless.DescriptorSet);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);

	for (const BlitCmd& bc : Impl_->Blits) {
		if (bc.Kind == BlitKind::Rgba) {
			if (!Impl_->BlitRgba.Pipeline) continue;
			BlitRgbaPc pc = bc.Rgba;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->BlitRgba.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OaU32 gx = (pc.dst_w + 7u) / 8u;
			OaU32 gy = (pc.dst_h + 7u) / 8u;
			vkCmdDispatch(cmd, gx, gy, 1);
		} else if (bc.Kind == BlitKind::Planar) {
			if (!Impl_->BlitPlanar.Pipeline) continue;
			BlitPlanarPc pc = bc.Planar;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->BlitPlanar.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OaU32 gx = (pc.dst_w + 7u) / 8u;
			OaU32 gy = (pc.dst_h + 7u) / 8u;
			vkCmdDispatch(cmd, gx, gy, 1);
		} else if (bc.Kind == BlitKind::ImageRgba) {
			if (!Impl_->BlitImageRgba.Pipeline || !bc.SrcImageView) continue;
			BlitImageRgbaPc pc = bc.ImageRgba;
			if (pc.src_idx == OA_BINDLESS_INVALID) continue;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->BlitImageRgba.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OaU32 gx = (pc.dst_w + 7u) / 8u;
			OaU32 gy = (pc.dst_h + 7u) / 8u;
			vkCmdDispatch(cmd, gx, gy, 1);
		} else if (bc.Kind == BlitKind::Rect) {
			if (!Impl_->DrawRect.Pipeline) continue;
			DrawRectPc pc = bc.Rect;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawRect.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OaU32 gx = (pc.dst_w + 7u) / 8u;
			OaU32 gy = (pc.dst_h + 7u) / 8u;
			vkCmdDispatch(cmd, gx, gy, 1);
		} else if (bc.Kind == BlitKind::RectOutline) {
			if (!Impl_->DrawRectOutline.Pipeline) continue;
			DrawRectOutlinePc pc = bc.RectOutline;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawRectOutline.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			OaU32 gx = (pc.dst_w + 7u) / 8u;
			OaU32 gy = (pc.dst_h + 7u) / 8u;
			vkCmdDispatch(cmd, gx, gy, 1);
		} else if (bc.Kind == BlitKind::RectOutlines) {
			if (!Impl_->DrawRectOutlines.Pipeline) continue;
			DrawRectOutlinesPc pc = bc.RectOutlines;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawRectOutlines.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, pc.count, 1, 1);
		} else if (bc.Kind == BlitKind::Line) {
			if (!Impl_->DrawLine.Pipeline) continue;
			DrawLinePc pc = bc.Line;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawLine.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, (pc.bounds_w + 7U) / 8U, (pc.bounds_h + 7U) / 8U, 1);
		} else if (bc.Kind == BlitKind::Glyphs) {
			if (!Impl_->DrawGlyphs.Pipeline) continue;
			DrawGlyphsPc pc = bc.Glyphs;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawGlyphs.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, pc.count, 1, 1);
		} else if (bc.Kind == BlitKind::Waveform) {
			if (!Impl_->DrawWaveform.Pipeline) continue;
			DrawWaveformPc pc = bc.Waveform;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawWaveform.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, (pc.dst_w + 63U) / 64U, 1, 1);
		} else if (bc.Kind == BlitKind::PlotLine) {
			if (!Impl_->DrawPlotLine.Pipeline) continue;
			DrawPlotLinePc pc = bc.PlotLine;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawPlotLine.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, (pc.dst_w + 63U) / 64U, 1, 1);
		} else if (bc.Kind == BlitKind::Heatmap) {
			if (!Impl_->DrawHeatmap.Pipeline) continue;
			DrawHeatmapPc pc = bc.Heatmap;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawHeatmap.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
				sizeof(pc), &pc);
			vkCmdDispatch(cmd, (pc.dst_w + 7U) / 8U,
				(pc.dst_h + 7U) / 8U, 1);
		}

		// Memory barrier between dispatches.
		VkMemoryBarrier mb{};
		mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &mb, 0, nullptr, 0, nullptr);
	}
}

void OaUi::EndFrame() {}

void OaUi::MarkFrameSubmitted(
	const OaVkTimelineSemaphore& InSemaphore,
	OaU64 InValue) {
	if (!Impl_ || InSemaphore.Semaphore == nullptr || InValue == 0U) return;
	for (const auto& used : Impl_->UsedPlots) {
		if (used.Cache >= Impl_->Plots.Size()
			|| used.Slot >= Impl::kPlotSlotCount) continue;
		Impl_->Plots[used.Cache].Slots[used.Slot].Completion =
			OaCompletionToken(Impl_->Rt->Device, InSemaphore, InValue);
	}
}


// ─── Style stack ─────────────────────────────────────────────────────────────

void OaUi::PushStyle(const OaUiStyle& InStyle) {
	if (!Impl_ || Impl_->StyleDepth >= Impl::kStyleDepth) return;
	Impl_->StyleStack[Impl_->StyleDepth++] = InStyle;
}

void OaUi::PopStyle() {
	if (!Impl_ || Impl_->StyleDepth == 0) return;
	--Impl_->StyleDepth;
}

const OaUiStyle& OaUi::CurrentStyle() const noexcept {
	if (!Impl_) {
		static const OaUiStyle sDefault;
		return sDefault;
	}
	if (Impl_->StyleDepth == 0) return Impl_->DefaultStyle;
	return Impl_->StyleStack[Impl_->StyleDepth - 1];
}


// ─── Layout containers ───────────────────────────────────────────────────────

void OaUi::BeginPanel(OaStringView /*InId*/, OaPixelRect InRect, const OaUiLayout& /*InLayout*/) {
	if (!Impl_) return;
	Impl::PanelState ps;
	ps.Rect   = InRect;
	ps.Cursor = 0.0F;
	Impl_->PanelStack.PushBack(ps);
}

void OaUi::EndPanel() {
	if (!Impl_ || Impl_->PanelStack.Empty()) return;
	Impl_->PanelStack.PopBack();
}

void OaUi::BeginRow(OaStringView /*InId*/) {}
void OaUi::EndRow() {}
void OaUi::Spacing(OaF32 /*InPixels*/) {}
void OaUi::Separator() {}


// ─── Widgets ──────────────────────────────────────────────────────────────────

namespace {

OaU32 HashWidgetId(OaStringView InId) noexcept {
	OaU32 hash = 2166136261U;
	for (OaUsize i = 0; i < InId.Size(); ++i) {
		hash ^= static_cast<OaU8>(InId[i]);
		hash *= 16777619U;
	}
	return hash == 0U ? 1U : hash;
}

} // namespace

bool OaUi::Button(OaStringView /*InLabel*/) { return false; }
bool OaUi::Checkbox(OaStringView /*InLabel*/, bool& /*InOutValue*/) { return false; }
bool OaUi::SliderF32(OaStringView /*InLabel*/, OaF32* /*InOutValue*/,
	OaF32 /*InMin*/, OaF32 /*InMax*/, const char* /*InFmt*/) { return false; }
bool OaUi::SliderI32(OaStringView /*InLabel*/, OaI32* /*InOutValue*/,
	OaI32 /*InMin*/, OaI32 /*InMax*/) { return false; }
bool OaUi::InputText(OaStringView /*InLabel*/, OaString& /*InOutText*/) { return false; }
bool OaUi::Dropdown(OaStringView /*InLabel*/, OaI32& /*InOutIndex*/,
	OaSpan<const OaStringView> /*InItems*/) { return false; }

void OaUi::Label(OaStringView /*InText*/) {}
void OaUi::LabelFmt(const char* /*InFmt*/, ...) {}
void OaUi::Text(OaStringView /*InText*/) {}
void OaUi::ColorSwatch(OaColor /*InColor*/, VlmVec2 /*InSize*/) {}
void OaUi::ProgressBar(OaF32 /*InFraction*/, OaStringView /*InOverlay*/) {}

bool OaUi::Timeline(
	OaStringView InId,
	OaPixelRect InRect,
	OaF32& InOutFraction) {
	if (!Impl_ || InRect.W <= 0 || InRect.H <= 0) return false;

	const OaU32 id = HashWidgetId(InId);
	const bool hovered = InRect.Contains(Input_.MouseX, Input_.MouseY);
	if (hovered) Input_.HoverId = id;
	if (hovered && Input_.LPressed) Input_.ActiveId = id;

	bool changed = false;
	if (Input_.ActiveId == id && (Input_.LButton || Input_.LReleased)) {
		const OaF32 next = std::clamp(
			(Input_.MouseX - static_cast<OaF32>(InRect.X))
				/ static_cast<OaF32>(InRect.W),
			0.0F,
			1.0F);
		changed = next != InOutFraction;
		InOutFraction = next;
	}
	if (Input_.ActiveId == id && Input_.LReleased) Input_.ActiveId = 0U;

	InOutFraction = std::clamp(InOutFraction, 0.0F, 1.0F);
	const OaUiStyle& style = CurrentStyle();
	Rect(InRect, style.Surface.WithAlpha(0.94F));
	const OaI32 fillWidth = static_cast<OaI32>(
		static_cast<OaF32>(InRect.W) * InOutFraction + 0.5F);
	if (fillWidth > 0) {
		Rect({InRect.X, InRect.Y, fillWidth, InRect.H}, style.Accent);
	}
	const OaI32 handleWidth = std::max<OaI32>(1,
		std::min<OaI32>(InRect.W, std::min<OaI32>(6, std::max<OaI32>(1, InRect.H / 2))));
	const OaI32 handleX = std::clamp(
		InRect.X + fillWidth - handleWidth / 2,
		InRect.X,
		InRect.X + InRect.W - handleWidth);
	Rect({handleX, InRect.Y - 3, handleWidth, InRect.H + 6},
		hovered || Input_.ActiveId == id ? style.AccentHover : style.TextSecondary);
	return changed;
}

bool OaUi::WaveformTimeline(
	OaStringView InId,
	OaPixelRect InRect,
	const OaMatrix& InEnvelope,
	OaF32& InOutFraction) {
	const OaMatrixShape shape = InEnvelope.GetShape();
	if (!Impl_ || InRect.X < 0 || InRect.Y < 0 || InRect.W <= 0 || InRect.H <= 0
		|| shape.Rank != 2 || shape[0] <= 0 || shape[1] != 2
		|| InEnvelope.GetDtype() != OaScalarType::Float32
		|| InEnvelope.HeapSlot() < 0) {
		return false;
	}

	const OaU32 id = HashWidgetId(InId);
	const bool hovered = InRect.Contains(Input_.MouseX, Input_.MouseY);
	if (hovered) Input_.HoverId = id;
	if (hovered && Input_.LPressed) Input_.ActiveId = id;

	bool changed = false;
	if (Input_.ActiveId == id && (Input_.LButton || Input_.LReleased)) {
		const OaF32 next = std::clamp(
			(Input_.MouseX - static_cast<OaF32>(InRect.X))
				/ static_cast<OaF32>(InRect.W),
			0.0F,
			1.0F);
		changed = next != InOutFraction;
		InOutFraction = next;
	}
	if (Input_.ActiveId == id && Input_.LReleased) Input_.ActiveId = 0U;
	InOutFraction = std::clamp(InOutFraction, 0.0F, 1.0F);

	const OaUiStyle& style = CurrentStyle();
	Rect(InRect, style.Surface.WithAlpha(0.55F));
	BlitCmd command{};
	command.Kind = BlitKind::Waveform;
	command.Waveform.envelope_idx = static_cast<OaU32>(InEnvelope.HeapSlot());
	command.Waveform.bins = static_cast<OaU32>(shape[0]);
	command.Waveform.dst_idx = 0U;
	command.Waveform.dst_x = static_cast<OaU32>(InRect.X);
	command.Waveform.dst_y = static_cast<OaU32>(InRect.Y);
	command.Waveform.dst_w = static_cast<OaU32>(InRect.W);
	command.Waveform.dst_h = static_cast<OaU32>(InRect.H);
	command.Waveform.fraction = InOutFraction;
	command.Waveform.played_rgba = style.Accent.ToU32();
	command.Waveform.remaining_rgba = style.TextSecondary.WithAlpha(0.60F).ToU32();
	Impl_->Blits.PushBack(command);

	const OaI32 playheadX = std::clamp(
		InRect.X + static_cast<OaI32>(static_cast<OaF32>(InRect.W) * InOutFraction),
		InRect.X,
		InRect.X + InRect.W - 1);
	Rect({playheadX, InRect.Y, hovered || Input_.ActiveId == id ? 3 : 2, InRect.H},
		hovered || Input_.ActiveId == id ? style.AccentHover : style.Text);
	return changed;
}


// ─── Image widgets ────────────────────────────────────────────────────────────

void OaUi::Image(OaU32 InBindlessIdx, OaI32 InW, OaI32 InH) {
	if (!Impl_) return;
	OaI32 dstX = 0, dstY = 0, dstW = InW, dstH = InH;
	if (!Impl_->PanelStack.Empty()) {
		const auto& ps = Impl_->PanelStack.Back();
		dstX = ps.Rect.X; dstY = ps.Rect.Y;
		dstW = ps.Rect.W; dstH = ps.Rect.H;
	}
	BlitCmd bc;
	bc.Kind         = BlitKind::Rgba;
	bc.Rgba.src_idx = InBindlessIdx;
	bc.Rgba.dst_idx = 0;
	bc.Rgba.src_w   = static_cast<OaU32>(InW);
	bc.Rgba.src_h   = static_cast<OaU32>(InH);
	bc.Rgba.dst_x   = static_cast<OaU32>(dstX);
	bc.Rgba.dst_y   = static_cast<OaU32>(dstY);
	bc.Rgba.dst_w   = static_cast<OaU32>(dstW);
	bc.Rgba.dst_h   = static_cast<OaU32>(dstH);
	Impl_->Blits.PushBack(bc);
}

void OaUi::ImageVkRgba(
	void* InImage,
	void* InImageView,
	OaI32 InW,
	OaI32 InH,
	VkImageLayout InLayout) {
	if (!Impl_ || !InImageView) return;
	OaI32 dstX = 0, dstY = 0, dstW = InW, dstH = InH;
	if (!Impl_->PanelStack.Empty()) {
		const auto& ps = Impl_->PanelStack.Back();
		dstX = ps.Rect.X; dstY = ps.Rect.Y;
		dstW = ps.Rect.W; dstH = ps.Rect.H;
	}
	BlitCmd bc;
	bc.Kind = BlitKind::ImageRgba;
	bc.SrcImage = static_cast<VkImage>(InImage);
	bc.SrcImageView = static_cast<VkImageView>(InImageView);
	bc.SrcImageLayout = InLayout;
	bc.ImageRgba.src_w = static_cast<OaU32>(InW);
	bc.ImageRgba.src_h = static_cast<OaU32>(InH);
	bc.ImageRgba.dst_idx = 0;
	bc.ImageRgba.dst_x = static_cast<OaU32>(dstX);
	bc.ImageRgba.dst_y = static_cast<OaU32>(dstY);
	bc.ImageRgba.dst_w = static_cast<OaU32>(dstW);
	bc.ImageRgba.dst_h = static_cast<OaU32>(dstH);
	Impl_->Blits.PushBack(bc);
}

void OaUi::Rect(OaPixelRect InRect, OaColor InColor) {
	if (!Impl_ || InRect.X < 0 || InRect.Y < 0 || InRect.W <= 0 || InRect.H <= 0) {
		return;
	}
	BlitCmd bc{};
	bc.Kind = BlitKind::Rect;
	bc.Rect.dst_idx = 0;
	bc.Rect.dst_x = static_cast<OaU32>(InRect.X);
	bc.Rect.dst_y = static_cast<OaU32>(InRect.Y);
	bc.Rect.dst_w = static_cast<OaU32>(InRect.W);
	bc.Rect.dst_h = static_cast<OaU32>(InRect.H);
	bc.Rect.rgba = InColor.ToU32();
	Impl_->Blits.PushBack(bc);
}

void OaUi::RectOutline(
	OaPixelRect InRect,
	OaColor InColor,
	OaU32 InThickness) {
	if (!Impl_ || InRect.X < 0 || InRect.Y < 0 || InRect.W <= 0 || InRect.H <= 0) {
		return;
	}
	BlitCmd bc{};
	bc.Kind = BlitKind::RectOutline;
	bc.RectOutline.dst_idx = 0;
	bc.RectOutline.dst_x = static_cast<OaU32>(InRect.X);
	bc.RectOutline.dst_y = static_cast<OaU32>(InRect.Y);
	bc.RectOutline.dst_w = static_cast<OaU32>(InRect.W);
	bc.RectOutline.dst_h = static_cast<OaU32>(InRect.H);
	bc.RectOutline.thickness = std::max<OaU32>(1, InThickness);
	bc.RectOutline.rgba = InColor.ToU32();
	Impl_->Blits.PushBack(bc);
}

void OaUi::Line(
	VlmVec2 InBegin,
	VlmVec2 InEnd,
	OaColor InColor,
	OaF32 InThickness) {
	if (!Impl_ || !std::isfinite(InBegin.X) || !std::isfinite(InBegin.Y)
		|| !std::isfinite(InEnd.X) || !std::isfinite(InEnd.Y)
		|| !std::isfinite(InThickness) || InThickness <= 0.0F) return;
	const OaF32 padding = InThickness * 0.5F + 1.5F;
	const OaI32 x = std::max<OaI32>(0, static_cast<OaI32>(
		std::floor(std::min(InBegin.X, InEnd.X) - padding)));
	const OaI32 y = std::max<OaI32>(0, static_cast<OaI32>(
		std::floor(std::min(InBegin.Y, InEnd.Y) - padding)));
	const OaI32 right = static_cast<OaI32>(
		std::ceil(std::max(InBegin.X, InEnd.X) + padding));
	const OaI32 bottom = static_cast<OaI32>(
		std::ceil(std::max(InBegin.Y, InEnd.Y) + padding));
	if (right <= x || bottom <= y) return;
	BlitCmd command{};
	command.Kind = BlitKind::Line;
	command.Line.dst_idx = 0U;
	command.Line.x0 = InBegin.X;
	command.Line.y0 = InBegin.Y;
	command.Line.x1 = InEnd.X;
	command.Line.y1 = InEnd.Y;
	command.Line.thickness = InThickness;
	command.Line.rgba = InColor.ToU32();
	command.Line.bounds_x = static_cast<OaU32>(x);
	command.Line.bounds_y = static_cast<OaU32>(y);
	command.Line.bounds_w = static_cast<OaU32>(right - x);
	command.Line.bounds_h = static_cast<OaU32>(bottom - y);
	Impl_->Blits.PushBack(command);
}

void OaUi::RectOutlines(
	const OaDetectionBuffer& InDetections,
	OaPixelRect InDstRect,
	OaPixelRect InClipRect,
	OaColor InColor,
	OaU32 InThickness) {
	if (!Impl_ || InDetections.Count() == 0
		|| InDetections.BindlessIndex() == OA_BINDLESS_INVALID
		|| InDstRect.W <= 0 || InDstRect.H <= 0
		|| InClipRect.W <= 0 || InClipRect.H <= 0) {
		return;
	}
	BlitCmd bc{};
	bc.Kind = BlitKind::RectOutlines;
	bc.RectOutlines.rect_idx = InDetections.BindlessIndex();
	bc.RectOutlines.count = InDetections.Count();
	bc.RectOutlines.dst_idx = 0;
	bc.RectOutlines.dst_x = InDstRect.X;
	bc.RectOutlines.dst_y = InDstRect.Y;
	bc.RectOutlines.dst_w = static_cast<OaU32>(InDstRect.W);
	bc.RectOutlines.dst_h = static_cast<OaU32>(InDstRect.H);
	bc.RectOutlines.clip_x = InClipRect.X;
	bc.RectOutlines.clip_y = InClipRect.Y;
	bc.RectOutlines.clip_w = static_cast<OaU32>(InClipRect.W);
	bc.RectOutlines.clip_h = static_cast<OaU32>(InClipRect.H);
	bc.RectOutlines.thickness = std::max<OaU32>(1, InThickness);
	bc.RectOutlines.rgba = InColor.ToU32();
	Impl_->Blits.PushBack(bc);
}

void OaUi::Glyphs(
	const OaGlyphBuffer& InGlyphs,
	const OaTextAtlas& InAtlas,
	OaPixelRect InDstRect,
	OaPixelRect InClipRect) {
	const OaU32 atlasIndex = InAtlas.AtlasBindlessIndex(OaFontId::Sans);
	if (!Impl_ || InGlyphs.Count() == 0
		|| InGlyphs.BindlessIndex() == OA_BINDLESS_INVALID
		|| atlasIndex == OA_BINDLESS_INVALID
		|| InDstRect.W <= 0 || InDstRect.H <= 0
		|| InClipRect.W <= 0 || InClipRect.H <= 0) {
		return;
	}
	BlitCmd bc{};
	bc.Kind = BlitKind::Glyphs;
	bc.Glyphs.glyph_idx = InGlyphs.BindlessIndex();
	bc.Glyphs.atlas_idx = atlasIndex;
	bc.Glyphs.count = InGlyphs.Count();
	bc.Glyphs.dst_idx = 0;
	bc.Glyphs.dst_x = InDstRect.X;
	bc.Glyphs.dst_y = InDstRect.Y;
	bc.Glyphs.dst_w = static_cast<OaU32>(InDstRect.W);
	bc.Glyphs.dst_h = static_cast<OaU32>(InDstRect.H);
	bc.Glyphs.clip_x = InClipRect.X;
	bc.Glyphs.clip_y = InClipRect.Y;
	bc.Glyphs.clip_w = static_cast<OaU32>(InClipRect.W);
	bc.Glyphs.clip_h = static_cast<OaU32>(InClipRect.H);
	bc.Glyphs.atlas_w = static_cast<OaU32>(InAtlas.AtlasWidth());
	bc.Glyphs.atlas_h = static_cast<OaU32>(InAtlas.AtlasHeight());
	bc.Glyphs.px_range = InAtlas.PxRange();
	Impl_->Blits.PushBack(bc);
}

void OaUi::ImagePlanar(const OaImagePlanes& InPlanes, OaI32 InDstX, OaI32 InDstY) {
	if (!Impl_) return;

	OaI32 dstX = InDstX, dstY = InDstY;
	OaI32 dstW = InPlanes.Width, dstH = InPlanes.Height;
	if (!Impl_->PanelStack.Empty()) {
		const auto& ps = Impl_->PanelStack.Back();
		dstX = ps.Rect.X + InDstX;
		dstY = ps.Rect.Y + InDstY;
		dstW = ps.Rect.W;
		dstH = ps.Rect.H;
	}

	OaU32 dtypes = 0;
	for (OaU32 c = 0; c < static_cast<OaU32>(InPlanes.ChannelCount); ++c) {
		dtypes |= (static_cast<OaU32>(InPlanes.Dtypes[c]) & 0x3U) << (c * 2U);
	}

	BlitCmd bc;
	bc.Kind            = BlitKind::Planar;
	bc.Planar.r_idx    = (InPlanes.ChannelCount > 0) ? InPlanes.Planes[0].BindlessIndex : OA_BINDLESS_INVALID;
	bc.Planar.g_idx    = (InPlanes.ChannelCount > 1) ? InPlanes.Planes[1].BindlessIndex : OA_BINDLESS_INVALID;
	bc.Planar.b_idx    = (InPlanes.ChannelCount > 2) ? InPlanes.Planes[2].BindlessIndex : OA_BINDLESS_INVALID;
	bc.Planar.a_idx    = (InPlanes.ChannelCount > 3) ? InPlanes.Planes[3].BindlessIndex : OA_BINDLESS_INVALID;
	bc.Planar.dst_idx  = 0;
	bc.Planar.src_w    = static_cast<OaU32>(InPlanes.Width);
	bc.Planar.src_h    = static_cast<OaU32>(InPlanes.Height);
	bc.Planar.dst_x    = static_cast<OaU32>(dstX);
	bc.Planar.dst_y    = static_cast<OaU32>(dstY);
	bc.Planar.dtypes   = dtypes;
	bc.Planar.channels = static_cast<OaU32>(InPlanes.ChannelCount);
	bc.Planar.dst_w    = static_cast<OaU32>(dstW);
	bc.Planar.dst_h    = static_cast<OaU32>(dstH);
	Impl_->Blits.PushBack(bc);
}


// ─── Data visualization ──────────────────────────────────────────────────────

void OaUi::PlotLine(
	OaStringView InLabel,
	const OaF32* InData,
	OaI32 InCount,
	const OaUiPlotConfig& InCfg) {
	if (!Impl_ || !Impl_->Rt || InData == nullptr || InCount <= 0
		|| Impl_->PanelStack.Empty()) return;
	const OaPixelRect rect = Impl_->PanelStack.Back().Rect;
	if (rect.X < 0 || rect.Y < 0 || rect.W <= 0 || rect.H <= 0) return;

	OaU32 cacheIndex = 0U;
	OaU32 slotIndex = 0U;
	auto* slot = Impl_->UploadPlotValues(HashWidgetId(InLabel), InData,
		static_cast<OaU32>(InCount), cacheIndex, slotIndex);
	if (slot == nullptr) return;
	OaF32 yMin = InCfg.YMin;
	OaF32 yMax = InCfg.YMax;
	if (InCfg.AutoScale) {
		yMin = std::numeric_limits<OaF32>::infinity();
		yMax = -std::numeric_limits<OaF32>::infinity();
		const auto* values = static_cast<const OaF32*>(slot->Buffer.MappedPtr);
		for (OaU32 index = 0; index < slot->Count; ++index) {
			if (!std::isfinite(values[index])) continue;
			yMin = std::min(yMin, values[index]);
			yMax = std::max(yMax, values[index]);
		}
	}
	if (!std::isfinite(yMin) || !std::isfinite(yMax)) return;
	if (yMax <= yMin) {
		const OaF32 margin = std::max(1.0e-4F, std::abs(yMin) * 0.05F);
		yMin -= margin;
		yMax += margin;
	}

	Rect(rect, CurrentStyle().Surface.WithAlpha(0.94F));
	BlitCmd command{};
	command.Kind = BlitKind::PlotLine;
	command.PlotLine.values_idx = slot->Buffer.BindlessIndex;
	command.PlotLine.count = slot->Count;
	command.PlotLine.dst_idx = 0U;
	command.PlotLine.dst_x = static_cast<OaU32>(rect.X);
	command.PlotLine.dst_y = static_cast<OaU32>(rect.Y);
	command.PlotLine.dst_w = static_cast<OaU32>(rect.W);
	command.PlotLine.dst_h = static_cast<OaU32>(rect.H);
	command.PlotLine.y_min = yMin;
	command.PlotLine.y_max = yMax;
	command.PlotLine.rgba = InCfg.Color.ToU32();
	command.PlotLine.flags = (InCfg.ShowGrid ? 1U : 0U)
		| (InCfg.Fill ? 2U : 0U);
	Impl_->Blits.PushBack(command);
	Impl_->UsedPlots.PushBack({.Cache = cacheIndex, .Slot = slotIndex});
}

void OaUi::PlotLineRing(
	OaStringView InLabel,
	const OaF32* InData,
	OaI32 InCount,
	OaI32 InOffset,
	const OaUiPlotConfig& InCfg) {
	if (InData == nullptr || InCount <= 0) return;
	std::vector<OaF32> ordered(static_cast<OaUsize>(InCount));
	const OaI32 offset = ((InOffset % InCount) + InCount) % InCount;
	for (OaI32 index = 0; index < InCount; ++index) {
		ordered[static_cast<OaUsize>(index)] =
			InData[static_cast<OaUsize>((offset + index) % InCount)];
	}
	PlotLine(InLabel, ordered.data(), InCount, InCfg);
}
void OaUi::Heatmap(OaStringView /*InLabel*/, const OaVkBuffer& InBuffer,
	const OaUiHeatmapConfig& InCfg) {
	if (!Impl_ || InBuffer.BindlessIndex == OA_BINDLESS_INVALID
		|| InCfg.Rows <= 0 || InCfg.Cols <= 0
		|| Impl_->PanelStack.Empty()) return;
	const OaPixelRect rect = Impl_->PanelStack.Back().Rect;
	if (rect.X < 0 || rect.Y < 0 || rect.W <= 0 || rect.H <= 0) return;
	OaF32 vMin = InCfg.VMin;
	OaF32 vMax = InCfg.VMax;
	if (!std::isfinite(vMin) || !std::isfinite(vMax)) return;
	if (vMax <= vMin) {
		const OaF32 margin = std::max(1.0e-4F, std::abs(vMin) * 0.05F);
		vMin -= margin;
		vMax += margin;
	}
	BlitCmd command{};
	command.Kind = BlitKind::Heatmap;
	command.Heatmap.values_idx = InBuffer.BindlessIndex;
	command.Heatmap.rows = static_cast<OaU32>(InCfg.Rows);
	command.Heatmap.cols = static_cast<OaU32>(InCfg.Cols);
	command.Heatmap.dst_idx = 0U;
	command.Heatmap.dst_x = static_cast<OaU32>(rect.X);
	command.Heatmap.dst_y = static_cast<OaU32>(rect.Y);
	command.Heatmap.dst_w = static_cast<OaU32>(rect.W);
	command.Heatmap.dst_h = static_cast<OaU32>(rect.H);
	command.Heatmap.v_min = vMin;
	command.Heatmap.v_max = vMax;
	command.Heatmap.colormap = std::min(InCfg.Colormap, 3U);
	command.Heatmap.value_type = std::min(InCfg.ValueType, 2U);
	command.Heatmap.offset_elements = InCfg.OffsetElements;
	command.Heatmap.flags = InCfg.ShowGrid ? 1U : 0U;
	Impl_->Blits.PushBack(command);
}

void OaUi::Heatmap(OaStringView InLabel, const OaMatrix& InMatrix,
	const OaUiHeatmapConfig& InCfg) {
	if (InMatrix.IsEmpty() || InMatrix.Rank() != 2) return;
	OaUiHeatmapConfig config = InCfg;
	if (config.Rows <= 0) config.Rows = static_cast<OaI32>(InMatrix.Size(0));
	if (config.Cols <= 0) config.Cols = static_cast<OaI32>(InMatrix.Size(1));
	switch (InMatrix.GetDtype()) {
		case OaScalarType::Float32: config.ValueType = 0U; break;
		case OaScalarType::UInt32: config.ValueType = 1U; break;
		case OaScalarType::Int32: config.ValueType = 2U; break;
		default: return;
	}
	config.OffsetElements += static_cast<OaU32>(InMatrix.ByteOffset_ / 4U);
	const OaVkBuffer buffer = InMatrix.GetVkBuffer();
	Heatmap(InLabel, buffer, config);
}

void OaUi::Heatmap(OaStringView InLabel, const OaF32* InData,
	OaI32 InRows, OaI32 InCols, const OaUiHeatmapConfig& InCfg) {
	if (!Impl_ || InData == nullptr || InRows <= 0 || InCols <= 0) return;
	const OaI64 count64 = static_cast<OaI64>(InRows) * InCols;
	if (count64 <= 0 || count64 > Impl::kPlotCapacity) return;
	OaU32 cacheIndex = 0U;
	OaU32 slotIndex = 0U;
	auto* slot = Impl_->UploadPlotValues(HashWidgetId(InLabel), InData,
		static_cast<OaU32>(count64), cacheIndex, slotIndex);
	if (slot == nullptr) return;
	OaUiHeatmapConfig config = InCfg;
	config.Rows = InRows;
	config.Cols = InCols;
	config.ValueType = 0U;
	config.OffsetElements = 0U;
	Heatmap(InLabel, slot->Buffer, config);
	Impl_->UsedPlots.PushBack({.Cache = cacheIndex, .Slot = slotIndex});
}
