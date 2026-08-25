#include <oa/runtime/app.h>
#include <oa/core/log.h>
#include "executionSession.h"
#include "presentationPlatform.h"
#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

class ActiveSessionRestore {
public:
	ActiveSessionRestore() noexcept
		: previous_(oa::ExecutionSession::getActivePtr())
	{}

	~ActiveSessionRestore() {
		oa::ExecutionSession::setActive(previous_);
	}

private:
	oa::ExecutionSession* previous_ = nullptr;
};

} // namespace

static void appInitConsole() {
#if defined(_WIN32)
	setConsoleCP(CP_UTF8);
	setConsoleOutputCP(CP_UTF8);
#endif
}

int oa::withEngine(
	oa::Fn<int(oa::Engine&)> inBody,
	const oa::EngineConfig& inConfig)
{
	appInitConsole();
	ActiveSessionRestore restoreSelection;
	if (not inBody) {
		OaLogError(oa::LogComponent::App,
			"oa::withEngine requires an application body");
		return 1;
	}

	auto engineResult = oa::Engine::create(inConfig);
	if (not engineResult) {
		OaLogError(oa::LogComponent::App,
			"Engine init failed: %s", engineResult.getStatus().toString().cStr());
		return 1;
	}
	auto engine = oa::move(*engineResult);

	const int bodyResult = inBody(*engine);
	// Destroy captured values while their engine is still alive. Ordinary locals
	// in the callback have already left scope; this also covers resource-owning
	// captures retained by oa::Fn itself.
	inBody = oa::Fn<int(oa::Engine&)>{};

	const auto closeStatus = engine->close();
	if (not closeStatus) {
		OaLogError(oa::LogComponent::App,
			"Engine close failed: %s", closeStatus.toString().cStr());
		return bodyResult == 0 ? 1 : bodyResult;
	}
	return bodyResult;
}

int oa::withEngine(
	oa::StringView inAppName,
	oa::Fn<int(oa::Engine&)> inBody,
	oa::PresentationMode inPresentationMode)
{
	oa::EngineConfig config;
	config.appName = oa::String(inAppName);
	config.presentationMode = inPresentationMode;
	config.preloadEmbeddedPipelines = false;
	oa::PresentationPlatformLease platform;
	if (inPresentationMode == oa::PresentationMode::Swapchain) {
		const oa::Status platformStatus = platform.acquire(&config);
		if (not platformStatus.isOk()) {
			OaLogError(oa::LogComponent::App,
				"Presentation platform init failed: %s",
				platformStatus.toString().cStr());
			return 1;
		}
	}
	return oa::withEngine(oa::move(inBody), config);
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
