// OaCv::Frame — implementation. See Source/Public/Oa/Vision/CvFrame.h.
//
// Thin adapter over OaCvFrame (Oa/Ui/Cv.h): accumulate overlays through a
// per-call API, then on Commit run OaCvFrame::Render to get a single RGBA8
// OaTexture and hand it to the bound OaPlot::Axes via Imshow. The axes is
// the consumer; the Frame's only job is to bridge the per-overlay API to
// the renderer-owned overlay variant types.

// Engine first — VK_NO_PROTOTYPES before any vulkan.h pull-in.
#include <Oa/Runtime/Engine.h>

#include <Oa/Vision/CvFrame.h>
#include <Oa/Core/Log.h>


namespace OaCv {

Frame::Frame(OaPlot::Axes& InAx, const OaTexture& InBase)
	: Ax_(&InAx)
	, Base_(InBase)
{
	Cv_.W    = InBase.Width;
	Cv_.H    = InBase.Height;
	// OaCvFrame::Base is a non-owning pointer to the source RGBA8 buffer.
	// OaTexture::DeviceBuf is that buffer; cast away the const since
	// OaCvFrame holds a non-const pointer (it does not mutate the buffer).
	Cv_.Base = const_cast<OaVkBuffer*>(&Base_.DeviceBuf);
}


Frame::Frame(Frame&& InOther) noexcept
	: Ax_(InOther.Ax_)
	, Base_(std::move(InOther.Base_))
	, Cv_(std::move(InOther.Cv_))
	, Overlay_(std::move(InOther.Overlay_))
{
	// Rebind to the new Base_ — the moved-from object's Base_ is destroyed.
	Cv_.Base       = const_cast<OaVkBuffer*>(&Base_.DeviceBuf);
	InOther.Ax_    = nullptr;
	InOther.Cv_.Base = nullptr;
}


Frame& Frame::operator=(Frame&& InOther) noexcept {
	if (this != &InOther) {
		Ax_       = InOther.Ax_;
		Base_     = std::move(InOther.Base_);
		Cv_       = std::move(InOther.Cv_);
		Overlay_  = std::move(InOther.Overlay_);
		Cv_.Base  = const_cast<OaVkBuffer*>(&Base_.DeviceBuf);
		InOther.Ax_      = nullptr;
		InOther.Cv_.Base = nullptr;
	}
	return *this;
}


void Frame::BBoxes(OaVec<BBox> InBoxes, const BBoxStyle& InStyle) {
	OaCvBboxesConfig cfg;
	cfg.Color      = InStyle.Color;
	cfg.LineWidth  = InStyle.LineWidth;
	cfg.Alpha      = InStyle.Alpha;
	cfg.ShowLabels = InStyle.ShowLabels;
	cfg.ShowScores = InStyle.ShowScores;
	Cv_.AddBboxes(std::move(InBoxes), cfg);
}


void Frame::Masks(OaVec<OaCvMask> InMasks, const OaCvMasksConfig& InCfg) {
	Cv_.AddMasks(std::move(InMasks), InCfg);
}


void Frame::Keypoints() {
	OA_LOG_WARN(OaLogComponent::App,
		"OaCv::Frame::Keypoints: not implemented yet (Phase-2)");
}


void Frame::Stats() {
	// OaCvFrame already has AddStats with a default config; the stats overlay
	// renders FPS / NumDetections / etc. in a configurable corner of the frame.
	Cv_.AddStats({});
}


OaStatus Frame::Commit(OaComputeEngine& InRt) {
	if (Ax_ == nullptr) {
		return OaStatus::Error("OaCv::Frame::Commit: no axes bound");
	}
	auto r = Cv_.Render(InRt);
	if (not r.IsOk()) { return r.GetStatus(); }
	if (Overlay_.IsValid()) { Overlay_.Destroy(InRt); }
	Overlay_ = *r;
	Ax_->Imshow(Overlay_);
	return OaStatus::Ok();
}


void Frame::Destroy(OaComputeEngine& InRt) {
	if (Overlay_.IsValid()) {
		Overlay_.Destroy(InRt);
	}
}

}  // namespace OaCv
