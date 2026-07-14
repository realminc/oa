#include <Oa/Vision/ItVideo.h>

OaResult<OaItVideo> OaItVideo::Create(
	OaEngine& InEngine, const OaItVideoConfig& InConfig)
{
	auto video = OaVideo::Create(InEngine, InConfig);
	if (not video.IsOk()) return video.GetStatus();
	OaItVideo iterator;
	iterator.Video_.Emplace(OaStdMove(*video));
	return iterator;
}

void OaItVideo::Destroy()
{
	if (Video_.HasValue()) Video_->Destroy();
	Video_.Reset();
}

bool OaItVideo::IsDone() const { return not Video_.HasValue() or Video_->IsDone(); }
void OaItVideo::Next() { if (Video_.HasValue()) (void)Video_->Next(); }
void OaItVideo::Reset() { if (Video_.HasValue()) Video_->Reset(); }
OaI64 OaItVideo::Index() const { return Video_.HasValue() ? Video_->Index() : 0; }
void OaItVideo::Play() { if (Video_.HasValue()) Video_->Play(); }
void OaItVideo::Pause() { if (Video_.HasValue()) Video_->Pause(); }
void OaItVideo::TogglePlay() { if (Video_.HasValue()) Video_->TogglePlay(); }
bool OaItVideo::IsPlaying() const { return Video_.HasValue() and Video_->IsPlaying(); }
OaStatus OaItVideo::StepForward() { return Video_.HasValue()
	? Video_->StepForward()
	: OaStatus::Error(OaStatusCode::FailedPrecondition, "OaItVideo is closed"); }
OaStatus OaItVideo::StepBackward() { return Video_.HasValue()
	? Video_->StepBackward()
	: OaStatus::Error(OaStatusCode::FailedPrecondition, "OaItVideo is closed"); }
OaStatus OaItVideo::StepFrames(OaI32 InFrameDelta) { return Video_.HasValue()
	? Video_->StepFrames(InFrameDelta)
	: OaStatus::Error(OaStatusCode::FailedPrecondition, "OaItVideo is closed"); }
OaStatus OaItVideo::Seek(OaU64 InTimestamp) { return Video_.HasValue()
	? Video_->Seek(InTimestamp)
	: OaStatus::Error(OaStatusCode::FailedPrecondition, "OaItVideo is closed"); }
OaStatus OaItVideo::Flush() { return Video_.HasValue() ? Video_->Flush() : OaStatus::Ok(); }
void OaItVideo::Tick(OaF32 InDeltaMs) { if (Video_.HasValue()) Video_->Tick(InDeltaMs); }
const OaVideoFrame& OaItVideo::CurrentFrame() const {
	static const OaVideoFrame empty = {};
	return Video_.HasValue() ? Video_->CurrentFrame() : empty;
}
OaResult<OaVec<OaU8>> OaItVideo::ReadbackCurrentRgba() { return Video_.HasValue()
	? Video_->ReadbackCurrentRgba()
	: OaResult<OaVec<OaU8>>(OaStatus::Error(
		OaStatusCode::FailedPrecondition, "OaItVideo is closed")); }
void OaItVideo::MarkCurrentFrameConsumed(
	const OaVkTimelineSemaphore& InSemaphore, OaU64 InValue)
{
	if (Video_.HasValue()) Video_->MarkCurrentFrameConsumed(InSemaphore, InValue);
}
OaU32 OaItVideo::Width() const { return Video_.HasValue() ? Video_->Width() : 0U; }
OaU32 OaItVideo::Height() const { return Video_.HasValue() ? Video_->Height() : 0U; }
OaU32 OaItVideo::FrameRate() const { return Video_.HasValue() ? Video_->FrameRate() : 0U; }
OaUsize OaItVideo::FrameCount() const { return Video_.HasValue() ? Video_->FrameCount() : 0U; }
OaF32 OaItVideo::FrameIntervalMs() const { return Video_.HasValue()
	? Video_->FrameIntervalMs() : 0.0F; }
bool OaItVideo::IsEos() const { return not Video_.HasValue() or Video_->IsEos(); }
const OaContainerInfo& OaItVideo::GetContainerInfo() const {
	static const OaContainerInfo empty = {};
	return Video_.HasValue() ? Video_->GetContainerInfo() : empty;
}
