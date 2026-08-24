#pragma once

#include <oa/ml/environment.h>
#include <oa/runtime/executionSession.h>

namespace oa {

class Engine;

// Private execution owner behind oa::Environment. It deliberately reuses the
// canonical oa::ExecutionSession implementation without exposing the private
// recorder through the high-level RL contract.
class EnvironmentExecution {
public:
	explicit EnvironmentExecution(oa::Engine& inEngine);
	~EnvironmentExecution();

	EnvironmentExecution(const EnvironmentExecution&) = delete;
	EnvironmentExecution& operator=(const EnvironmentExecution&) = delete;

	[[nodiscard]] oa::Status begin();
	[[nodiscard]] oa::Result<oa::Event> submit(bool& outAccepted);
	[[nodiscard]] oa::Status wait(const oa::Event& inEvent);
	[[nodiscard]] oa::Status cancel();
	[[nodiscard]] oa::Status close();

	[[nodiscard]] oa::ExecutionSession& session() const noexcept;
	[[nodiscard]] bool isOpen() const noexcept;
	[[nodiscard]] bool hasActiveRecording() const noexcept;
	[[nodiscard]] bool hasPendingEvent() const noexcept;
	[[nodiscard]] oa::U64 submissionCount() const noexcept {
		return submissionCount_;
	}

private:
	enum class State : oa::U8 {
		Ready,
		Recording,
		Submitted,
		Closed,
	};

	oa::Engine* engine_ = nullptr;
	oa::UniquePtr<oa::ExecutionSession> session_;
	oa::Event pendingEvent_;
	State state_ = State::Closed;
	oa::U64 submissionCount_ = 0;
};

// Internal RAII selector used by environment implementations, collectors and
// evaluators. No public header names oa::ExecutionSession.
class EnvironmentRecordingScope {
public:
	explicit EnvironmentRecordingScope(oa::Environment& inEnvironment);
	explicit EnvironmentRecordingScope(
		const oa::Environment& inEnvironment);

	EnvironmentRecordingScope(const EnvironmentRecordingScope&) = delete;
	EnvironmentRecordingScope& operator=(const EnvironmentRecordingScope&) = delete;

private:
	oa::ExecutionSession::RecordingScope scope_;
};

class EnvironmentExecutionAccess {
public:
	[[nodiscard]] static oa::ExecutionSession& session(oa::Environment& inEnvironment) noexcept;
	[[nodiscard]] static oa::ExecutionSession& session(
		const oa::Environment& inEnvironment) noexcept;
};

} // namespace oa
