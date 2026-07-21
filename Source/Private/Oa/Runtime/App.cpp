#include <Oa/Runtime/App.h>
#include <Oa/Core/Log.h>
#if defined(_WIN32)
#include <windows.h>
#endif

static void OaAppInitConsole() {
#if defined(_WIN32)
	SetConsoleCP(CP_UTF8);
	SetConsoleOutputCP(CP_UTF8);
#endif
}

int OaComputeApp::Main(int argc, char** argv) {
	OaAppInitConsole();
#ifdef NDEBUG
	OA_LOG_INIT("", OaLogLevel::Info, true);
#else
	OA_LOG_INIT("", OaLogLevel::Debug, true);
#endif

	int setupResult = Setup(argc, argv);
	if (setupResult not_eq 0 or not IsRunning) {
		return setupResult;
	}

	EngineConfig_.Extensions = Extensions_;
	// Initialize the pinned engine (RtStorage_) in place — no build-then-move.
	auto engineStatus = RtStorage_->InitInPlace(EngineConfig_);
	if (not engineStatus) {
		OA_LOG_ERROR(OaLogComponent::Core, "Engine init failed: %s", engineStatus.ToString().c_str());
		auto closeStatus = Rt.Close();
		if (not closeStatus) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Engine close after init failure failed: %s",
				closeStatus.ToString().c_str());
		}
		return 1;
	}

	int exitCode = 0;
	auto initStatus = Init();
	if (not initStatus) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"App init failed: %s", initStatus.ToString().c_str());
		exitCode = 1;
	} else {
		while (IsRunning) {
			auto tickStatus = Tick();
			if (not tickStatus) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"Tick error: %s", tickStatus.ToString().c_str());
				exitCode = 1;
				break;
			}
		}
	}

	Shutdown();
	auto closeStatus = Rt.Close();
	if (not closeStatus) {
		OA_LOG_ERROR(OaLogComponent::Core,
			"Engine close failed: %s", closeStatus.ToString().c_str());
		exitCode = 1;
	}
	return exitCode;
}
