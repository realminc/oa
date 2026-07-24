#include <Oa/Runtime/App.h>

#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"
#include "../../OaTest.h"

enum class OaComputeAppProbeMode : OaU8 {
	Success,
	InitFailure,
	InitStops,
	TickFailure,
	InvalidDevice,
	RetirementFailure,
};

class OaComputeAppProbeState {
public:
	OaVec<OaString> Calls_;
	OaU32 SetupCount_ = 0;
	OaU32 InitCount_ = 0;
	OaU32 TickCount_ = 0;
	OaU32 ShutdownCount_ = 0;
	OaU32 RetirementCompleteCount_ = 0;
	OaU32 RetirementReleaseCount_ = 0;
	OaBool ShutdownSawReady_ = false;
	OaBool RetirementCompleteSawDestroying_ = false;
};

class OaFailingRetirementProbe {
public:
	OaEngine* Engine_ = nullptr;
	OaSharedPtr<OaComputeAppProbeState> State_;

	[[nodiscard]] static OaStatus Complete(void* InPayload) {
		auto* payload = static_cast<OaFailingRetirementProbe*>(InPayload);
		++payload->State_->RetirementCompleteCount_;
		payload->State_->Calls_.PushBack("RetirementComplete");
		payload->State_->RetirementCompleteSawDestroying_ =
			payload->Engine_->GetState() == OaEngineState::Destroying;
		if (payload->State_->RetirementCompleteCount_ == 1) {
			return OaStatus::Error(
				OaStatusCode::Internal,
				"injected retirement completion failure");
		}
		return OaStatus::Ok();
	}

	static void Release(void* InPayload) {
		OaUniquePtr<OaFailingRetirementProbe> payload(
			static_cast<OaFailingRetirementProbe*>(InPayload));
		++payload->State_->RetirementReleaseCount_;
		payload->State_->Calls_.PushBack("RetirementRelease");
	}
};

class OaComputeAppProbe final : public OaComputeApp {
public:
	explicit OaComputeAppProbe(
		OaComputeAppProbeMode InMode = OaComputeAppProbeMode::Success,
		int InSetupResult = 0)
		: Mode_(InMode)
		, SetupResult_(InSetupResult)
	{}

	[[nodiscard]] OaEngineState EngineState() const noexcept {
		return Rt.GetState();
	}
	[[nodiscard]] OaStatus RetryClose() { return Rt.Close(); }
	[[nodiscard]] const OaVec<OaString>& Calls() const noexcept {
		return State_->Calls_;
	}
	[[nodiscard]] OaU32 SetupCount() const noexcept { return State_->SetupCount_; }
	[[nodiscard]] OaU32 InitCount() const noexcept { return State_->InitCount_; }
	[[nodiscard]] OaU32 TickCount() const noexcept { return State_->TickCount_; }
	[[nodiscard]] OaU32 ShutdownCount() const noexcept { return State_->ShutdownCount_; }
	[[nodiscard]] OaU32 RetirementCompleteCount() const noexcept {
		return State_->RetirementCompleteCount_;
	}
	[[nodiscard]] OaU32 RetirementReleaseCount() const noexcept {
		return State_->RetirementReleaseCount_;
	}
	[[nodiscard]] OaBool ShutdownSawReady() const noexcept {
		return State_->ShutdownSawReady_;
	}
	[[nodiscard]] OaBool RetirementCompleteSawDestroying() const noexcept {
		return State_->RetirementCompleteSawDestroying_;
	}

protected:
	int Setup(int InArgc, char** InArgv) override {
		(void)InArgc;
		(void)InArgv;
		++State_->SetupCount_;
		State_->Calls_.PushBack("Setup");

		EngineConfig_ = OaTestEngineConfig(OaPrecision::FP32);
		EngineConfig_.AppName = "OaComputeAppTest";
		EngineConfig_.RegisterAsGlobal = false;
		EngineConfig_.PreloadEmbeddedPipelines = false;
		EngineConfig_.EnablePipelineCache = false;
		if (Mode_ == OaComputeAppProbeMode::InvalidDevice) {
			EngineConfig_.DevicePref = OaDevicePreference::ByIndex;
			EngineConfig_.DeviceIndex = 0xFFFFFFFEU;
		}
		return SetupResult_;
	}

	OaStatus Init() override {
		++State_->InitCount_;
		State_->Calls_.PushBack("Init");
		if (Mode_ == OaComputeAppProbeMode::RetirementFailure) {
			auto payload = OaMakeUniquePtr<OaFailingRetirementProbe>();
			payload->Engine_ = &Rt;
			payload->State_ = State_;
			OaBorrowedServiceRetirement::Retire(
				Rt,
				payload.Release(),
				&OaFailingRetirementProbe::Complete,
				&OaFailingRetirementProbe::Release);
		}
		if (Mode_ == OaComputeAppProbeMode::InitFailure) {
			return OaStatus::Error(OaStatusCode::Internal, "injected app init failure");
		}
		if (Mode_ == OaComputeAppProbeMode::InitStops) {
			IsRunning = false;
		}
		return OaStatus::Ok();
	}

	OaStatus Tick() override {
		++State_->TickCount_;
		State_->Calls_.PushBack("Tick");
		if (Mode_ == OaComputeAppProbeMode::TickFailure) {
			return OaStatus::Error(OaStatusCode::Internal, "injected tick failure");
		}
		IsRunning = false;
		return OaStatus::Ok();
	}

	void Shutdown() override {
		++State_->ShutdownCount_;
		State_->Calls_.PushBack("Shutdown");
		State_->ShutdownSawReady_ = Rt.GetState() == OaEngineState::Ready;
	}

private:
	OaSharedPtr<OaComputeAppProbeState> State_ =
		OaMakeSharedPtr<OaComputeAppProbeState>();
	OaComputeAppProbeMode Mode_ = OaComputeAppProbeMode::Success;
	int SetupResult_ = 0;
};

TEST(ComputeApp, SetupFailurePreservesExitCodeWithoutCreatingEngine) {
	OaComputeAppProbe app(OaComputeAppProbeMode::Success, 23);

	EXPECT_EQ(app.Main(0, nullptr), 23);
	EXPECT_EQ(app.EngineState(), OaEngineState::Empty);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 0U);
	EXPECT_EQ(app.TickCount(), 0U);
	EXPECT_EQ(app.ShutdownCount(), 0U);
	ASSERT_EQ(app.Calls().Size(), 1U);
	EXPECT_EQ(app.Calls()[0], "Setup");
}

TEST(ComputeApp, SetupStopPreservesZeroWithoutCreatingEngine) {
	OaComputeAppProbe app;
	app.IsRunning = false;

	EXPECT_EQ(app.Main(0, nullptr), 0);
	EXPECT_EQ(app.EngineState(), OaEngineState::Empty);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 0U);
	EXPECT_EQ(app.TickCount(), 0U);
	EXPECT_EQ(app.ShutdownCount(), 0U);
	ASSERT_EQ(app.Calls().Size(), 1U);
	EXPECT_EQ(app.Calls()[0], "Setup");
}

TEST(ComputeApp, SuccessfulTickShutsDownThenClosesEngine) {
	OaComputeAppProbe app;

	EXPECT_EQ(app.Main(0, nullptr), 0);
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 1U);
	EXPECT_EQ(app.TickCount(), 1U);
	EXPECT_EQ(app.ShutdownCount(), 1U);
	EXPECT_TRUE(app.ShutdownSawReady());
	ASSERT_EQ(app.Calls().Size(), 4U);
	EXPECT_EQ(app.Calls()[0], "Setup");
	EXPECT_EQ(app.Calls()[1], "Init");
	EXPECT_EQ(app.Calls()[2], "Tick");
	EXPECT_EQ(app.Calls()[3], "Shutdown");
}

TEST(ComputeApp, InitFailureStillShutsDownAndClosesEngine) {
	OaComputeAppProbe app(OaComputeAppProbeMode::InitFailure);

	EXPECT_EQ(app.Main(0, nullptr), 1);
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 1U);
	EXPECT_EQ(app.TickCount(), 0U);
	EXPECT_EQ(app.ShutdownCount(), 1U);
	EXPECT_TRUE(app.ShutdownSawReady());
	ASSERT_EQ(app.Calls().Size(), 3U);
	EXPECT_EQ(app.Calls()[0], "Setup");
	EXPECT_EQ(app.Calls()[1], "Init");
	EXPECT_EQ(app.Calls()[2], "Shutdown");
}

TEST(ComputeApp, InitCanStopBeforeTickAndStillShutsDownAndClosesEngine) {
	OaComputeAppProbe app(OaComputeAppProbeMode::InitStops);

	EXPECT_EQ(app.Main(0, nullptr), 0);
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 1U);
	EXPECT_EQ(app.TickCount(), 0U);
	EXPECT_EQ(app.ShutdownCount(), 1U);
	EXPECT_TRUE(app.ShutdownSawReady());
	ASSERT_EQ(app.Calls().Size(), 3U);
	EXPECT_EQ(app.Calls()[0], "Setup");
	EXPECT_EQ(app.Calls()[1], "Init");
	EXPECT_EQ(app.Calls()[2], "Shutdown");
}

TEST(ComputeApp, TickFailureReturnsFailureAfterShutdownAndClose) {
	OaComputeAppProbe app(OaComputeAppProbeMode::TickFailure);

	EXPECT_EQ(app.Main(0, nullptr), 1);
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 1U);
	EXPECT_EQ(app.TickCount(), 1U);
	EXPECT_EQ(app.ShutdownCount(), 1U);
	EXPECT_TRUE(app.ShutdownSawReady());
	ASSERT_EQ(app.Calls().Size(), 4U);
	EXPECT_EQ(app.Calls()[0], "Setup");
	EXPECT_EQ(app.Calls()[1], "Init");
	EXPECT_EQ(app.Calls()[2], "Tick");
	EXPECT_EQ(app.Calls()[3], "Shutdown");
}

TEST(ComputeApp, EngineInitFailureClosesWithoutEnteringAppLifecycle) {
	OaComputeAppProbe app(OaComputeAppProbeMode::InvalidDevice);

	EXPECT_EQ(app.Main(0, nullptr), 1);
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.SetupCount(), 1U);
	EXPECT_EQ(app.InitCount(), 0U);
	EXPECT_EQ(app.TickCount(), 0U);
	EXPECT_EQ(app.ShutdownCount(), 0U);
	ASSERT_EQ(app.Calls().Size(), 1U);
	EXPECT_EQ(app.Calls()[0], "Setup");
}

TEST(ComputeApp, EngineCloseFailureRetainsBorrowedServiceForSuccessfulRetry) {
	OaComputeAppProbe app(OaComputeAppProbeMode::RetirementFailure);

	EXPECT_EQ(app.Main(0, nullptr), 1);
	EXPECT_EQ(app.EngineState(), OaEngineState::Failed);
	EXPECT_EQ(app.ShutdownCount(), 1U);
	EXPECT_EQ(app.RetirementCompleteCount(), 1U);
	EXPECT_EQ(app.RetirementReleaseCount(), 0U);
	EXPECT_TRUE(app.ShutdownSawReady());
	EXPECT_TRUE(app.RetirementCompleteSawDestroying());
	ASSERT_EQ(app.Calls().Size(), 5U);
	EXPECT_EQ(app.Calls()[0], "Setup");
	EXPECT_EQ(app.Calls()[1], "Init");
	EXPECT_EQ(app.Calls()[2], "Tick");
	EXPECT_EQ(app.Calls()[3], "Shutdown");
	EXPECT_EQ(app.Calls()[4], "RetirementComplete");

	ASSERT_TRUE(app.RetryClose().IsOk());
	EXPECT_EQ(app.EngineState(), OaEngineState::Destroyed);
	EXPECT_EQ(app.RetirementCompleteCount(), 2U);
	EXPECT_EQ(app.RetirementReleaseCount(), 1U);
	ASSERT_EQ(app.Calls().Size(), 7U);
	EXPECT_EQ(app.Calls()[5], "RetirementComplete");
	EXPECT_EQ(app.Calls()[6], "RetirementRelease");
}
