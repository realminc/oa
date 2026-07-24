#pragma once

#include <Oa/Ml/Rl/Environment.h>
#include <Oa/Runtime/Context.h>

class OaEngine;

// Private execution owner behind OaRlEnvironment. It deliberately reuses the
// canonical OaContext/OaExecutionSession implementation without exposing that
// compatibility facade through the high-level RL contract.
class OaRlEnvironmentExecution {
public:
	explicit OaRlEnvironmentExecution(OaEngine& InEngine);
	~OaRlEnvironmentExecution();

	OaRlEnvironmentExecution(const OaRlEnvironmentExecution&) = delete;
	OaRlEnvironmentExecution& operator=(const OaRlEnvironmentExecution&) = delete;

	[[nodiscard]] OaStatus Begin();
	[[nodiscard]] OaResult<OaEvent> Submit(bool& OutAccepted);
	[[nodiscard]] OaStatus Wait(const OaEvent& InEvent);
	[[nodiscard]] OaStatus Cancel();
	[[nodiscard]] OaStatus Close();

	[[nodiscard]] OaContext& Context() const noexcept;
	[[nodiscard]] bool IsOpen() const noexcept;
	[[nodiscard]] bool HasActiveRecording() const noexcept;
	[[nodiscard]] bool HasPendingEvent() const noexcept;
	[[nodiscard]] OaU64 SubmissionCount() const noexcept {
		return SubmissionCount_;
	}

private:
	enum class State : OaU8 {
		Ready,
		Recording,
		Submitted,
		Closed,
	};

	OaEngine* Engine_ = nullptr;
	OaUniquePtr<OaContext> Context_;
	OaEvent PendingEvent_;
	State State_ = State::Closed;
	OaU64 SubmissionCount_ = 0;
};

// Internal RAII selector used by environment implementations, collectors and
// evaluators. No public header names OaContext.
class OaRlEnvironmentRecordingScope {
public:
	explicit OaRlEnvironmentRecordingScope(OaRlEnvironment& InEnvironment);
	explicit OaRlEnvironmentRecordingScope(
		const OaRlEnvironment& InEnvironment);

	OaRlEnvironmentRecordingScope(const OaRlEnvironmentRecordingScope&) = delete;
	OaRlEnvironmentRecordingScope& operator=(const OaRlEnvironmentRecordingScope&) = delete;

private:
	OaContext::RecordingScope Scope_;
};

class OaRlEnvironmentExecutionAccess {
public:
	[[nodiscard]] static OaContext& Context(OaRlEnvironment& InEnvironment) noexcept;
	[[nodiscard]] static OaContext& Context(
		const OaRlEnvironment& InEnvironment) noexcept;
};
