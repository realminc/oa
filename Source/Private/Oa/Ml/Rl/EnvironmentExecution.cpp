#include "EnvironmentExecution.h"

#include <Oa/Runtime/Engine.h>

#include <cassert>

OaRlEnvironmentExecution::OaRlEnvironmentExecution(OaEngine& InEngine)
	: Engine_(&InEngine) {
	if (InEngine.IsReady()) {
		Context_.Reset(OaContext::Create(&InEngine));
		State_ = State::Ready;
	}
}

OaRlEnvironmentExecution::~OaRlEnvironmentExecution() {
	// Destruction is abandonment, never completion. Cancel an unsubmitted
	// recording; OaContext transfers an incomplete submitted batch to the
	// engine retirement queue without waiting.
	if (State_ == State::Recording && Context_) {
		Context_->Clear();
	}
	Context_.Reset();
	PendingEvent_ = {};
	State_ = State::Closed;
}

OaStatus OaRlEnvironmentExecution::Begin() {
	if (!Context_ || !Engine_ || !Engine_->IsReady()
		|| State_ == State::Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment execution session is closed");
	}
	if (State_ == State::Submitted) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment execution requires Wait before another recording");
	}
	State_ = State::Recording;
	return OaStatus::Ok();
}

OaResult<OaEvent> OaRlEnvironmentExecution::Submit(bool& OutAccepted) {
	OutAccepted = false;
	if (!Context_ || State_ != State::Recording) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment Submit requires an active recording");
	}
	auto completion = Context_->Submit();
	if (completion.IsError()) {
		const OaStatus failure = completion.GetStatus();
		OutAccepted = Context_->HasPendingSubmission();
		// Context submission may have acquired/recorded a private batch before
		// failing. Abandon that context so its stream is cancelled or retired;
		// a caller must record the complete transaction again rather than retry
		// Submit against an empty/partially recovered batch.
		Context_.Reset();
		PendingEvent_ = {};
		if (!OutAccepted && Engine_ && Engine_->IsReady()) {
			Context_.Reset(OaContext::Create(Engine_));
			State_ = State::Ready;
		} else {
			// Accepted work without a valid exact completion is retired by the
			// context destructor. The environment cannot safely record again.
			State_ = State::Closed;
		}
		return failure;
	}
	if (!completion->IsValid()) {
		const OaStatus failure = OaStatus::Error(
			OaStatusCode::Internal,
			"RL environment submission returned an invalid event");
		OutAccepted = Context_->HasPendingSubmission();
		Context_.Reset();
		PendingEvent_ = {};
		if (!OutAccepted && Engine_ && Engine_->IsReady()) {
			Context_.Reset(OaContext::Create(Engine_));
			State_ = State::Ready;
		} else {
			State_ = State::Closed;
		}
		return failure;
	}
	PendingEvent_ = *completion;
	State_ = State::Submitted;
	++SubmissionCount_;
	OutAccepted = true;
	return *completion;
}

OaStatus OaRlEnvironmentExecution::Wait(const OaEvent& InEvent) {
	if (!Context_ || State_ != State::Submitted
		|| !PendingEvent_.IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment Wait requires a submitted event");
	}
	const OaStatus status = Context_->Wait(InEvent);
	if (status.IsError()) return status;
	PendingEvent_ = {};
	State_ = State::Ready;
	return OaStatus::Ok();
}

OaStatus OaRlEnvironmentExecution::Cancel() {
	if (!Context_ || State_ == State::Closed) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment execution session is closed");
	}
	if (State_ == State::Submitted) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment cannot cancel submitted GPU work");
	}
	if (State_ == State::Recording) Context_->Clear();
	State_ = State::Ready;
	return OaStatus::Ok();
}

OaStatus OaRlEnvironmentExecution::Close() {
	if (State_ == State::Closed) return OaStatus::Ok();
	if (State_ == State::Submitted) {
		OA_RETURN_IF_ERROR(Wait(PendingEvent_));
	} else if (State_ == State::Recording && Context_) {
		Context_->Clear();
	}
	Context_.Reset();
	PendingEvent_ = {};
	State_ = State::Closed;
	return OaStatus::Ok();
}

OaContext& OaRlEnvironmentExecution::Context() const noexcept {
	assert(Context_ && "RL environment execution context is closed");
	return *Context_;
}

bool OaRlEnvironmentExecution::IsOpen() const noexcept {
	return State_ != State::Closed;
}

bool OaRlEnvironmentExecution::HasActiveRecording() const noexcept {
	return State_ == State::Recording;
}

bool OaRlEnvironmentExecution::HasPendingEvent() const noexcept {
	return State_ == State::Submitted && PendingEvent_.IsValid();
}

OaRlEnvironmentRecordingScope::OaRlEnvironmentRecordingScope(
	OaRlEnvironment& InEnvironment)
	: Scope_(InEnvironment.Execution_->Context()) {}

OaRlEnvironmentRecordingScope::OaRlEnvironmentRecordingScope(
	const OaRlEnvironment& InEnvironment)
	: Scope_(InEnvironment.Execution_->Context()) {}

OaContext& OaRlEnvironmentExecutionAccess::Context(
	OaRlEnvironment& InEnvironment) noexcept {
	return InEnvironment.Execution_->Context();
}

OaContext& OaRlEnvironmentExecutionAccess::Context(
	const OaRlEnvironment& InEnvironment) noexcept {
	return InEnvironment.Execution_->Context();
}

OaRlEnvironment::OaRlEnvironment(OaEngine& InEngine)
	: Execution_(OaMakeUniquePtr<OaRlEnvironmentExecution>(InEngine)) {}

OaRlEnvironment::~OaRlEnvironment() = default;
OaRlEnvironment::OaRlEnvironment(OaRlEnvironment&&) noexcept = default;
OaRlEnvironment& OaRlEnvironment::operator=(OaRlEnvironment&&) noexcept = default;

OaStatus OaRlEnvironment::Begin() {
	return Execution_ ? Execution_->Begin() : OaStatus::Error(
		OaStatusCode::FailedPrecondition,
		"RL environment has no execution session");
}

OaStatus OaRlEnvironment::RecordCommands(
	const std::function<OaStatus()>& InCommands) {
	if (!InCommands) {
		return OaStatus::InvalidArgument(
			"RL environment command callback must not be empty");
	}
	OA_RETURN_IF_ERROR(Begin());
	OaStatus status;
	{
		OaRlEnvironmentRecordingScope scope(*this);
		status = InCommands();
	}
	if (status.IsError()) (void)Cancel();
	return status;
}

OaStatus OaRlEnvironment::ResetEnvironment(OaU64 InSeed) {
	OA_RETURN_IF_ERROR(Begin());
	OaStatus status;
	{
		OaRlEnvironmentRecordingScope scope(*this);
		status = RecordResetEnvironment_(InSeed);
	}
	if (status.IsError()) (void)Cancel();
	return status;
}

OaResult<OaRlEnvironmentTransition> OaRlEnvironment::StepEnvironment(
	const OaMatrix& InAction) {
	const OaStatus begin = Begin();
	if (begin.IsError()) return begin;
	OaResult<OaRlEnvironmentTransition> result = OaStatus::Error(
		OaStatusCode::Internal, "RL environment step did not record");
	{
		OaRlEnvironmentRecordingScope scope(*this);
		result = RecordStepEnvironment_(InAction);
	}
	if (result.IsError()) (void)Cancel();
	return result;
}

OaStatus OaRlEnvironment::ResetCompleted() {
	OA_RETURN_IF_ERROR(Begin());
	OaStatus status;
	{
		OaRlEnvironmentRecordingScope scope(*this);
		status = RecordResetCompleted_();
	}
	if (status.IsError()) (void)Cancel();
	return status;
}

OaResult<OaEvent> OaRlEnvironment::Submit() {
	if (!Execution_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment has no execution session");
	}
	bool accepted = false;
	auto completion = Execution_->Submit(accepted);
	if (accepted) CommitRecordedState_();
	else if (completion.IsError()) RollbackRecordedState_();
	return completion;
}

OaStatus OaRlEnvironment::Wait(const OaEvent& InEvent) {
	return Execution_ ? Execution_->Wait(InEvent) : OaStatus::Error(
		OaStatusCode::FailedPrecondition,
		"RL environment has no execution session");
}

OaStatus OaRlEnvironment::Cancel() {
	if (!Execution_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"RL environment has no execution session");
	}
	const OaStatus status = Execution_->Cancel();
	if (status.IsOk()) RollbackRecordedState_();
	return status;
}

OaStatus OaRlEnvironment::Close() {
	if (!Execution_) return OaStatus::Ok();
	const OaStatus status = Execution_->Close();
	if (status.IsOk()) RollbackRecordedState_();
	return status;
}

bool OaRlEnvironment::IsOpen() const noexcept {
	return Execution_ && Execution_->IsOpen();
}

bool OaRlEnvironment::HasActiveRecording() const noexcept {
	return Execution_ && Execution_->HasActiveRecording();
}

bool OaRlEnvironment::HasPendingEvent() const noexcept {
	return Execution_ && Execution_->HasPendingEvent();
}

OaU64 OaRlEnvironment::SubmissionCount() const noexcept {
	return Execution_ ? Execution_->SubmissionCount() : 0U;
}
