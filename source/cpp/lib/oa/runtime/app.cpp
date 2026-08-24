#include <oa/runtime/app.h>
#include <oa/core/log.h>
#if defined(_WIN32)
#include <windows.h>
#endif

static void appInitConsole() {
#if defined(_WIN32)
	setConsoleCP(CP_UTF8);
	setConsoleOutputCP(CP_UTF8);
#endif
}

int oa::ComputeApp::main(int argc, char** argv) {
	appInitConsole();
	if (engineOwner_ != nullptr) {
		OaLogError(oa::LogComponent::App,
			"oa::ComputeApp::main is one-shot after engine creation");
		return 1;
	}

	int setupResult = setup(argc, argv);
	if (setupResult not_eq 0 or not isRunning) {
		return setupResult;
	}

	auto engineResult = oa::Engine::create(engineConfig_);
	if (not engineResult) {
		OaLogError(oa::LogComponent::App,
			"Engine init failed: %s", engineResult.getStatus().toString().cStr());
		return 1;
	}
	engineOwner_ = oa::move(*engineResult);

	int exitCode = 0;
	auto initStatus = init();
	if (not initStatus) {
		OaLogError(oa::LogComponent::App,
			"App init failed: %s", initStatus.toString().cStr());
		exitCode = 1;
	} else {
		while (isRunning) {
			auto tickStatus = tick();
			if (not tickStatus) {
				OaLogError(oa::LogComponent::App,
					"tick error: %s", tickStatus.toString().cStr());
				exitCode = 1;
				break;
			}
		}
	}

	shutdown();
	auto closeStatus = engine().close();
	if (not closeStatus) {
		OaLogError(oa::LogComponent::App,
			"Engine close failed: %s", closeStatus.toString().cStr());
		exitCode = 1;
	}
	return exitCode;
}
