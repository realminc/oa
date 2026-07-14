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
#include <Oa/Vision/Detection.h>

#include <cstring>

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

// ─── DrawRectOutline push constants (must match DrawRectOutline.slang) ───────
struct DrawRectOutlinePc {
	OaU32 dst_idx;
	OaU32 dst_x;
	OaU32 dst_y;
	OaU32 dst_w;
	OaU32 dst_h;
	OaU32 thickness;
	OaU32 rgba;
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
	RectOutline,
	RectOutlines,
	Glyphs,
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
		DrawRectOutlinePc RectOutline;
		DrawRectOutlinesPc RectOutlines;
		DrawGlyphsPc Glyphs;
	};
};


// ─── OaUi::Impl ────────────────────────────────────────────────────────────────

struct OaUi::Impl {
	OaComputeEngine* Rt = nullptr;

	// Blit pipelines (created in InitBlit).
	OaComputePipeline BlitRgba;
	OaComputePipeline BlitPlanar;
	OaComputePipeline BlitImageRgba;
	OaComputePipeline DrawRectOutline;
	OaComputePipeline DrawRectOutlines;
	OaComputePipeline DrawGlyphs;

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
		for (const auto& slot : Impl_->SampledImageSlots) {
			if (slot.Slot != OA_BINDLESS_INVALID) {
				Impl_->Rt->Bindless.DeregisterSampledImage(slot.Slot);
			}
		}
		Impl_->SampledImageSlots.Clear();
		Impl_->BlitRgba.Destroy(Impl_->Rt->Device);
		Impl_->BlitPlanar.Destroy(Impl_->Rt->Device);
		Impl_->BlitImageRgba.Destroy(Impl_->Rt->Device);
		Impl_->DrawRectOutline.Destroy(Impl_->Rt->Device);
		Impl_->DrawRectOutlines.Destroy(Impl_->Rt->Device);
		Impl_->DrawGlyphs.Destroy(Impl_->Rt->Device);
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
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutline", Impl_->DrawRectOutline));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawRectOutlines", Impl_->DrawRectOutlines));
	OA_RETURN_IF_ERROR(createBlitPipeline("DrawGlyphs", Impl_->DrawGlyphs));
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
	Impl_->PanelStack.Clear();
	Impl_->StyleDepth = 0;
}

bool OaUi::RouteEvent(const OaUiEvent& InEvent) {
	Input_.MouseX   = InEvent.MouseX;
	Input_.MouseY   = InEvent.MouseY;
	return false;
}

void OaUi::RecordRender(VkCommandBuffer InCmd, OaU32 InDstBindlessIdx) {
	if (!Impl_ || Impl_->Blits.Empty()) return;
	if (!Impl_->BlitRgba.Pipeline && !Impl_->BlitPlanar.Pipeline
		&& !Impl_->BlitImageRgba.Pipeline && !Impl_->DrawRectOutline.Pipeline
		&& !Impl_->DrawRectOutlines.Pipeline && !Impl_->DrawGlyphs.Pipeline) {
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
		} else {
			if (!Impl_->DrawGlyphs.Pipeline) continue;
			DrawGlyphsPc pc = bc.Glyphs;
			pc.dst_idx = InDstBindlessIdx;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				static_cast<VkPipeline>(Impl_->DrawGlyphs.Pipeline));
			vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, pc.count, 1, 1);
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
	if (!Impl_ || Impl_->StyleDepth == 0) {
		static const OaUiStyle sDefault;
		return sDefault;
	}
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


// ─── Widgets — all stub ───────────────────────────────────────────────────────

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


// ─── Data visualization — all stub ───────────────────────────────────────────

void OaUi::PlotLine(OaStringView /*InLabel*/, const OaF32* /*InData*/,
	OaI32 /*InCount*/, const OaUiPlotConfig& /*InCfg*/) {}
void OaUi::PlotLineRing(OaStringView /*InLabel*/, const OaF32* /*InData*/,
	OaI32 /*InCount*/, OaI32 /*InOffset*/, const OaUiPlotConfig& /*InCfg*/) {}
void OaUi::PlotHistogram(OaStringView /*InLabel*/, const OaF32* /*InData*/,
	OaI32 /*InCount*/, const OaUiPlotConfig& /*InCfg*/) {}
void OaUi::Heatmap(OaStringView /*InLabel*/, const OaVkBuffer& /*InBuffer*/,
	const OaUiHeatmapConfig& /*InCfg*/) {}
