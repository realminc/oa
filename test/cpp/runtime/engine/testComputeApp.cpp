#include <oa/runtime/app.h>

#include "oa/runtime/engine/borrowedServiceRetirement.h"
#include "../../oaTest.h"

namespace oa {

enum class ComputeAppProbeMode : oa::U8 {
	Success,
	InitFailure,
	InitStops,
	TickFailure,
	InvalidDevice,
	RetirementFailure,
};

class ComputeAppProbeState {
public:
	oa::Vec<oa::String> calls_;
	oa::U32 setupCount_ = 0;
	oa::U32 initCount_ = 0;
	oa::U32 tickCount_ = 0;
	oa::U32 shutdownCount_ = 0;
	oa::U32 retirementCompleteCount_ = 0;
	oa::U32 retirementReleaseCount_ = 0;
	oa::Bool shutdownSawReady_ = false;
	oa::Bool retirementCompleteSawDestroying_ = false;
};

class FailingRetirementProbe {
public:
	oa::Engine* engine_ = nullptr;
	oa::SharedPtr<oa::ComputeAppProbeState> state_;

	[[nodiscard]] static oa::Status complete(void* inPayload) {
		auto* payload = static_cast<FailingRetirementProbe*>(inPayload);
		++payload->state_->retirementCompleteCount_;
		payload->state_->calls_.pushBack("RetirementComplete");
		payload->state_->retirementCompleteSawDestroying_ =
			payload->engine_->getState() == oa::EngineState::Destroying;
		if (payload->state_->retirementCompleteCount_ == 1) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"injected retirement completion failure");
		}
		return oa::Status::ok();
	}

	static void release(void* inPayload) {
		oa::UniquePtr<FailingRetirementProbe> payload(
			static_cast<FailingRetirementProbe*>(inPayload));
		++payload->state_->retirementReleaseCount_;
		payload->state_->calls_.pushBack("RetirementRelease");
	}
};

class ComputeAppProbe final : public oa::ComputeApp {
public:
	explicit ComputeAppProbe(
		oa::ComputeAppProbeMode inMode = oa::ComputeAppProbeMode::Success,
		int inSetupResult = 0)
		: mode_(inMode)
		, setupResult_(inSetupResult)
	{}

	[[nodiscard]] oa::EngineState engineState() const noexcept {
		return hasEngine() ? engine().getState() : oa::EngineState::Empty;
	}
	[[nodiscard]] oa::Bool ownsEngine() const noexcept { return hasEngine(); }
	[[nodiscard]] oa::Status retryClose() { return engine().close(); }
	[[nodiscard]] const oa::Vec<oa::String>& calls() const noexcept {
		return state_->calls_;
	}
	[[nodiscard]] oa::U32 setupCount() const noexcept { return state_->setupCount_; }
	[[nodiscard]] oa::U32 initCount() const noexcept { return state_->initCount_; }
	[[nodiscard]] oa::U32 tickCount() const noexcept { return state_->tickCount_; }
	[[nodiscard]] oa::U32 shutdownCount() const noexcept { return state_->shutdownCount_; }
	[[nodiscard]] oa::U32 retirementCompleteCount() const noexcept {
		return state_->retirementCompleteCount_;
	}
	[[nodiscard]] oa::U32 retirementReleaseCount() const noexcept {
		return state_->retirementReleaseCount_;
	}
	[[nodiscard]] oa::Bool shutdownSawReady() const noexcept {
		return state_->shutdownSawReady_;
	}
	[[nodiscard]] oa::Bool retirementCompleteSawDestroying() const noexcept {
		return state_->retirementCompleteSawDestroying_;
	}

protected:
	int setup(int inArgc, char** inArgv) override {
		(void)inArgc;
		(void)inArgv;
		++state_->setupCount_;
		state_->calls_.pushBack("Setup");

		engineConfig_ = testEngineConfig(oa::Precision::FP32);
		engineConfig_.appName = "oa::ComputeAppTest";
		engineConfig_.selectForThread = false;
		engineConfig_.preloadEmbeddedPipelines = false;
		engineConfig_.enablePipelineCache = false;
		if (mode_ == oa::ComputeAppProbeMode::InvalidDevice) {
			engineConfig_.devicePref = oa::DevicePreference::ByIndex;
			engineConfig_.deviceIndex = 0xFFFFFFFEU;
		}
		return setupResult_;
	}

	oa::Status init() override {
		++state_->initCount_;
		state_->calls_.pushBack("init");
		if (mode_ == oa::ComputeAppProbeMode::RetirementFailure) {
			auto payload = oa::makeUnique<FailingRetirementProbe>();
			payload->engine_ = &engine();
			payload->state_ = state_;
			oa::BorrowedServiceRetirement::retire(
				engine(),
				payload.release(),
				&FailingRetirementProbe::complete,
				&FailingRetirementProbe::release);
		}
		if (mode_ == oa::ComputeAppProbeMode::InitFailure) {
			return oa::Status::error(oa::StatusCode::Internal, "injected app init failure");
		}
		if (mode_ == oa::ComputeAppProbeMode::InitStops) {
			isRunning = false;
		}
		return oa::Status::ok();
	}

	oa::Status tick() override {
		++state_->tickCount_;
		state_->calls_.pushBack("tick");
		if (mode_ == oa::ComputeAppProbeMode::TickFailure) {
			return oa::Status::error(oa::StatusCode::Internal, "injected tick failure");
		}
		isRunning = false;
		return oa::Status::ok();
	}

	void shutdown() override {
		++state_->shutdownCount_;
		state_->calls_.pushBack("shutdown");
		state_->shutdownSawReady_ = engine().getState() == oa::EngineState::Ready;
	}

private:
	oa::SharedPtr<oa::ComputeAppProbeState> state_ =
		oa::makeShared<oa::ComputeAppProbeState>();
	oa::ComputeAppProbeMode mode_ = oa::ComputeAppProbeMode::Success;
	int setupResult_ = 0;
};

} // namespace oa

TEST(WithEngine, OwnsOneLexicalEngineAndRestoresPreviousSelection) {
	auto config = testEngineConfig(oa::Precision::FP32);
	config.appName = "oa::withEngineTest";
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto* previous = oa::ExecutionSession::getActivePtr();
	oa::Engine* borrowed = nullptr;

	const int result = oa::withEngine([&](oa::Engine& engine) {
		borrowed = &engine;
		EXPECT_TRUE(engine.isReady());
		EXPECT_EQ(&oa::ExecutionSession::getActive().engine(), &engine);
		return 23;
	}, config);

	EXPECT_EQ(result, 23);
	EXPECT_NE(borrowed, nullptr);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
}

TEST(WithEngine, RejectsEmptyBodyWithoutChangingSelection) {
	auto* previous = oa::ExecutionSession::getActivePtr();
	EXPECT_EQ(oa::withEngine(oa::Fn<int(oa::Engine&)>{}), 1);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
}

TEST(WithEngine, EngineCreationFailureDoesNotInvokeBody) {
	auto config = testEngineConfig(oa::Precision::FP32);
	config.devicePref = oa::DevicePreference::ByIndex;
	config.deviceIndex = 0xFFFFFFFEU;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	bool invoked = false;
	auto* previous = oa::ExecutionSession::getActivePtr();

	EXPECT_EQ(oa::withEngine([&](oa::Engine&) {
		invoked = true;
		return 0;
	}, config), 1);
	EXPECT_FALSE(invoked);
	EXPECT_EQ(oa::ExecutionSession::getActivePtr(), previous);
}

TEST(ComputeApp, SetupFailurePreservesExitCodeWithoutCreatingEngine) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::Success, 23);

	EXPECT_EQ(app.main(0, nullptr), 23);
	EXPECT_EQ(app.engineState(), oa::EngineState::Empty);
	EXPECT_FALSE(app.ownsEngine());
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 0U);
	EXPECT_EQ(app.tickCount(), 0U);
	EXPECT_EQ(app.shutdownCount(), 0U);
	ASSERT_EQ(app.calls().size(), 1U);
	EXPECT_EQ(app.calls()[0], "Setup");
}

TEST(ComputeApp, SetupStopPreservesZeroWithoutCreatingEngine) {
	oa::ComputeAppProbe app;
	app.isRunning = false;

	EXPECT_EQ(app.main(0, nullptr), 0);
	EXPECT_EQ(app.engineState(), oa::EngineState::Empty);
	EXPECT_FALSE(app.ownsEngine());
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 0U);
	EXPECT_EQ(app.tickCount(), 0U);
	EXPECT_EQ(app.shutdownCount(), 0U);
	ASSERT_EQ(app.calls().size(), 1U);
	EXPECT_EQ(app.calls()[0], "Setup");
}

TEST(ComputeApp, SuccessfulTickShutsDownThenClosesEngine) {
	oa::ComputeAppProbe app;

	EXPECT_EQ(app.main(0, nullptr), 0);
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_TRUE(app.ownsEngine());
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 1U);
	EXPECT_EQ(app.tickCount(), 1U);
	EXPECT_EQ(app.shutdownCount(), 1U);
	EXPECT_TRUE(app.shutdownSawReady());
	ASSERT_EQ(app.calls().size(), 4U);
	EXPECT_EQ(app.calls()[0], "Setup");
	EXPECT_EQ(app.calls()[1], "init");
	EXPECT_EQ(app.calls()[2], "tick");
	EXPECT_EQ(app.calls()[3], "shutdown");
}

TEST(ComputeApp, MainCannotRestartAfterEngineCreation) {
	oa::ComputeAppProbe app;

	ASSERT_EQ(app.main(0, nullptr), 0);
	EXPECT_EQ(app.main(0, nullptr), 1);
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 1U);
	EXPECT_EQ(app.tickCount(), 1U);
	EXPECT_EQ(app.shutdownCount(), 1U);
}

TEST(ComputeApp, InitFailureStillShutsDownAndClosesEngine) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::InitFailure);

	EXPECT_EQ(app.main(0, nullptr), 1);
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_TRUE(app.ownsEngine());
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 1U);
	EXPECT_EQ(app.tickCount(), 0U);
	EXPECT_EQ(app.shutdownCount(), 1U);
	EXPECT_TRUE(app.shutdownSawReady());
	ASSERT_EQ(app.calls().size(), 3U);
	EXPECT_EQ(app.calls()[0], "Setup");
	EXPECT_EQ(app.calls()[1], "init");
	EXPECT_EQ(app.calls()[2], "shutdown");
}

TEST(ComputeApp, InitCanStopBeforeTickAndStillShutsDownAndClosesEngine) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::InitStops);

	EXPECT_EQ(app.main(0, nullptr), 0);
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 1U);
	EXPECT_EQ(app.tickCount(), 0U);
	EXPECT_EQ(app.shutdownCount(), 1U);
	EXPECT_TRUE(app.shutdownSawReady());
	ASSERT_EQ(app.calls().size(), 3U);
	EXPECT_EQ(app.calls()[0], "Setup");
	EXPECT_EQ(app.calls()[1], "init");
	EXPECT_EQ(app.calls()[2], "shutdown");
}

TEST(ComputeApp, TickFailureReturnsFailureAfterShutdownAndClose) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::TickFailure);

	EXPECT_EQ(app.main(0, nullptr), 1);
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 1U);
	EXPECT_EQ(app.tickCount(), 1U);
	EXPECT_EQ(app.shutdownCount(), 1U);
	EXPECT_TRUE(app.shutdownSawReady());
	ASSERT_EQ(app.calls().size(), 4U);
	EXPECT_EQ(app.calls()[0], "Setup");
	EXPECT_EQ(app.calls()[1], "init");
	EXPECT_EQ(app.calls()[2], "tick");
	EXPECT_EQ(app.calls()[3], "shutdown");
}

TEST(ComputeApp, EngineInitFailureClosesWithoutEnteringAppLifecycle) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::InvalidDevice);

	EXPECT_EQ(app.main(0, nullptr), 1);
	EXPECT_EQ(app.engineState(), oa::EngineState::Empty);
	EXPECT_FALSE(app.ownsEngine());
	EXPECT_EQ(app.setupCount(), 1U);
	EXPECT_EQ(app.initCount(), 0U);
	EXPECT_EQ(app.tickCount(), 0U);
	EXPECT_EQ(app.shutdownCount(), 0U);
	ASSERT_EQ(app.calls().size(), 1U);
	EXPECT_EQ(app.calls()[0], "Setup");
}

TEST(ComputeApp, EngineCloseFailureRetainsBorrowedServiceForSuccessfulRetry) {
	oa::ComputeAppProbe app(oa::ComputeAppProbeMode::RetirementFailure);

	EXPECT_EQ(app.main(0, nullptr), 1);
	EXPECT_EQ(app.engineState(), oa::EngineState::Failed);
	EXPECT_EQ(app.shutdownCount(), 1U);
	EXPECT_EQ(app.retirementCompleteCount(), 1U);
	EXPECT_EQ(app.retirementReleaseCount(), 0U);
	EXPECT_TRUE(app.shutdownSawReady());
	EXPECT_TRUE(app.retirementCompleteSawDestroying());
	ASSERT_EQ(app.calls().size(), 5U);
	EXPECT_EQ(app.calls()[0], "Setup");
	EXPECT_EQ(app.calls()[1], "init");
	EXPECT_EQ(app.calls()[2], "tick");
	EXPECT_EQ(app.calls()[3], "shutdown");
	EXPECT_EQ(app.calls()[4], "RetirementComplete");

	ASSERT_TRUE(app.retryClose().isOk());
	EXPECT_EQ(app.engineState(), oa::EngineState::Destroyed);
	EXPECT_EQ(app.retirementCompleteCount(), 2U);
	EXPECT_EQ(app.retirementReleaseCount(), 1U);
	ASSERT_EQ(app.calls().size(), 7U);
	EXPECT_EQ(app.calls()[5], "RetirementComplete");
	EXPECT_EQ(app.calls()[6], "RetirementRelease");
}
