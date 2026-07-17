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
	if (setupResult != 0 || !IsRunning) {
		return setupResult;
	}

	EngineConfig_.Extensions = Extensions_;
	// Initialize the pinned engine (RtStorage_) in place — no build-then-move.
	auto engineStatus = RtStorage_->InitInPlace(EngineConfig_);
	if (!engineStatus) {
		OA_LOG_ERROR(OaLogComponent::Core, "Engine init failed: %s", engineStatus.ToString().c_str());
		return 1;
	}

	auto initStatus = Init();
	if (!initStatus) {
		OA_LOG_ERROR(OaLogComponent::Core, "App init failed: %s",	initStatus.ToString().c_str());
		return 1;
	}

	while (IsRunning) {
		auto status = Tick();
		if (!status) {
			OA_LOG_ERROR(OaLogComponent::Core, "Tick error: %s", status.ToString().c_str());
			break;
		}
	}

	Shutdown();
	return 0;
}
