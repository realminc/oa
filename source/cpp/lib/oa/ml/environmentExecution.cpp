#include "environmentExecution.h"

#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <cassert>

oa::EnvironmentExecution::EnvironmentExecution(oa::Engine& inEngine)
	: engine_(&inEngine) {
	if (inEngine.isReady()) {
		session_ = oa::makeUnique<oa::ExecutionSession>(&inEngine);
		state_ = State::Ready;
	}
}

oa::EnvironmentExecution::~EnvironmentExecution() {
	// Destruction is abandonment, never completion. cancel an unsubmitted
	// recording; oa::ExecutionSession transfers an incomplete submitted batch to the
	// engine retirement queue without waiting.
	if (state_ == State::Recording && session_) {
		session_->clear();
	}
	session_.reset();
	pendingEvent_ = {};
	state_ = State::Closed;
}

oa::Status oa::EnvironmentExecution::begin() {
	if (!session_ || !engine_ || !engine_->isReady()
		|| state_ == State::Closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment execution session is closed");
	}
	if (state_ == State::Submitted) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment execution requires wait before another recording");
	}
	state_ = State::Recording;
	return oa::Status::ok();
}

oa::Result<oa::Event> oa::EnvironmentExecution::submit(bool& outAccepted) {
	outAccepted = false;
	if (!session_ || state_ != State::Recording) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment submit requires an active recording");
	}
	auto completion = session_->submit();
	if (completion.isError()) {
		const oa::Status failure = completion.getStatus();
		outAccepted = session_->hasPendingSubmission();
		// Session submission may have acquired/recorded a private batch before
		// failing. Abandon it so its stream is cancelled or retired;
		// a caller must record the complete transaction again rather than retry
		// submit against an empty/partially recovered batch.
		session_.reset();
		pendingEvent_ = {};
		if (!outAccepted && engine_ && engine_->isReady()) {
			session_ = oa::makeUnique<oa::ExecutionSession>(engine_);
			state_ = State::Ready;
		} else {
			// Accepted work without a valid exact completion is retired by the
			// session destructor. The environment cannot safely record again.
			state_ = State::Closed;
		}
		return failure;
	}
	if (!completion->isValid()) {
		const oa::Status failure = oa::Status::error(
			oa::StatusCode::Internal,
			"RL environment submission returned an invalid event");
		outAccepted = session_->hasPendingSubmission();
		session_.reset();
		pendingEvent_ = {};
		if (!outAccepted && engine_ && engine_->isReady()) {
			session_ = oa::makeUnique<oa::ExecutionSession>(engine_);
			state_ = State::Ready;
		} else {
			state_ = State::Closed;
		}
		return failure;
	}
	pendingEvent_ = *completion;
	state_ = State::Submitted;
	++submissionCount_;
	outAccepted = true;
	return *completion;
}

oa::Status oa::EnvironmentExecution::wait(const oa::Event& inEvent) {
	if (!session_ || state_ != State::Submitted
		|| !pendingEvent_.isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment wait requires a submitted event");
	}
	const oa::Status status = session_->wait(inEvent);
	if (status.isError()) return status;
	pendingEvent_ = {};
	state_ = State::Ready;
	return oa::Status::ok();
}

oa::Status oa::EnvironmentExecution::cancel() {
	if (!session_ || state_ == State::Closed) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment execution session is closed");
	}
	if (state_ == State::Submitted) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment cannot cancel submitted GPU work");
	}
	if (state_ == State::Recording) session_->clear();
	state_ = State::Ready;
	return oa::Status::ok();
}

oa::Status oa::EnvironmentExecution::close() {
	if (state_ == State::Closed) return oa::Status::ok();
	if (state_ == State::Submitted) {
		OA_RETURN_IF_ERROR(wait(pendingEvent_));
	} else if (state_ == State::Recording && session_) {
		session_->clear();
	}
	session_.reset();
	pendingEvent_ = {};
	state_ = State::Closed;
	return oa::Status::ok();
}

oa::ExecutionSession& oa::EnvironmentExecution::session() const noexcept {
	assert(session_ && "RL environment execution session is closed");
	return *session_;
}

bool oa::EnvironmentExecution::isOpen() const noexcept {
	return state_ != State::Closed;
}

bool oa::EnvironmentExecution::hasActiveRecording() const noexcept {
	return state_ == State::Recording;
}

bool oa::EnvironmentExecution::hasPendingEvent() const noexcept {
	return state_ == State::Submitted && pendingEvent_.isValid();
}

oa::EnvironmentRecordingScope::EnvironmentRecordingScope(
	oa::Environment& inEnvironment)
	: scope_(inEnvironment.execution_->session()) {}

oa::EnvironmentRecordingScope::EnvironmentRecordingScope(
	const oa::Environment& inEnvironment)
	: scope_(inEnvironment.execution_->session()) {}

oa::ExecutionSession& oa::EnvironmentExecutionAccess::session(
	oa::Environment& inEnvironment) noexcept {
	return inEnvironment.execution_->session();
}

oa::ExecutionSession& oa::EnvironmentExecutionAccess::session(
	const oa::Environment& inEnvironment) noexcept {
	return inEnvironment.execution_->session();
}

oa::Environment::Environment(oa::Engine& inEngine)
	: execution_(oa::makeUnique<oa::EnvironmentExecution>(inEngine)) {}

oa::Environment::~Environment() = default;
oa::Environment::Environment(oa::Environment&&) noexcept = default;
oa::Environment& oa::Environment::operator=(oa::Environment&&) noexcept = default;

oa::Status oa::Environment::begin() {
	return execution_ ? execution_->begin() : oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"RL environment has no execution session");
}

oa::Status oa::Environment::recordCommands(
	const std::function<oa::Status()>& inCommands) {
	if (!inCommands) {
		return oa::Status::invalidArgument(
			"RL environment command callback must not be empty");
	}
	OA_RETURN_IF_ERROR(begin());
	oa::Status status;
	{
		oa::EnvironmentRecordingScope scope(*this);
		status = inCommands();
	}
	if (status.isError()) (void)cancel();
	return status;
}

oa::Status oa::Environment::reset(oa::U64 inSeed) {
	OA_RETURN_IF_ERROR(begin());
	oa::Status status;
	{
		oa::EnvironmentRecordingScope scope(*this);
		status = recordReset_(inSeed);
	}
	if (status.isError()) (void)cancel();
	return status;
}

oa::Result<oa::EnvironmentTransition> oa::Environment::step(
	const oa::Matrix& inAction) {
	const oa::Status beginStatus = begin();
	if (beginStatus.isError()) return beginStatus;
	oa::Result<oa::EnvironmentTransition> result = oa::Status::error(
		oa::StatusCode::Internal, "RL environment step did not record");
	{
		oa::EnvironmentRecordingScope scope(*this);
		result = recordStep_(inAction);
	}
	if (result.isError()) (void)cancel();
	return result;
}

oa::Status oa::Environment::resetCompleted() {
	OA_RETURN_IF_ERROR(begin());
	oa::Status status;
	{
		oa::EnvironmentRecordingScope scope(*this);
		status = recordResetCompleted_();
	}
	if (status.isError()) (void)cancel();
	return status;
}

oa::Result<oa::Event> oa::Environment::submit() {
	if (!execution_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment has no execution session");
	}
	bool accepted = false;
	auto completion = execution_->submit(accepted);
	if (accepted) commitRecordedState_();
	else if (completion.isError()) rollbackRecordedState_();
	return completion;
}

oa::Status oa::Environment::wait(const oa::Event& inEvent) {
	return execution_ ? execution_->wait(inEvent) : oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"RL environment has no execution session");
}

oa::Status oa::Environment::cancel() {
	if (!execution_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"RL environment has no execution session");
	}
	const oa::Status status = execution_->cancel();
	if (status.isOk()) rollbackRecordedState_();
	return status;
}

oa::Status oa::Environment::close() {
	if (!execution_) return oa::Status::ok();
	const oa::Status status = execution_->close();
	if (status.isOk()) rollbackRecordedState_();
	return status;
}

bool oa::Environment::isOpen() const noexcept {
	return execution_ && execution_->isOpen();
}

bool oa::Environment::hasActiveRecording() const noexcept {
	return execution_ && execution_->hasActiveRecording();
}

bool oa::Environment::hasPendingEvent() const noexcept {
	return execution_ && execution_->hasPendingEvent();
}

oa::U64 oa::Environment::submissionCount() const noexcept {
	return execution_ ? execution_->submissionCount() : 0U;
}
