// oa::ComputeApp — compute apps with tick loop: oa::Engine + setup/init/tick/shutdown.
// Windowed apps compose oa::Presenter with the same engine; there is no graphics
// application subclass mirroring the removed engine hierarchy.
//
// usage:
//   class MyApp : public oa::ComputeApp {
//   public:
//       int setup(int argc, char** argv) override { /* parse args */ return 0; }
//       oa::Status tick() override { /* do work */ isRunning = false; return oa::Status::ok(); }
//   };
//   int main(int argc, char** argv) { MyApp app; return app.main(argc, argv); }

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/engine.h>

namespace oa {

// Run one bounded application body with one factory-created Engine. The engine
// is owned only for the lexical duration of this call and is passed to the body
// as a borrow. withEngine never stores a process-global owner and never submits
// or waits implicitly; the body still reaches an explicit submit/wait or checked
// sink boundary. Engine creation or close failure returns 1, otherwise the
// body's exit code is preserved.
[[nodiscard]] int withEngine(
	oa::Fn<int(oa::Engine&)> inBody,
	const oa::EngineConfig& inConfig = {});
[[nodiscard]] int withEngine(
	oa::StringView inAppName,
	oa::Fn<int(oa::Engine&)> inBody,
	oa::PresentationMode inPresentationMode = oa::PresentationMode::None);

// Vulkan application — owns one factory-created, pinned oa::Engine around the
// tick loop.
// Subclass for training, inference, compute apps.
// Device selection: engineConfig_.devicePref / deviceIndex (see oa::EngineConfig).
class ComputeApp {
public:
	virtual ~ComputeApp() = default;

	bool isRunning = true;

	// Orchestrates: Setup -> Engine::Create -> init -> tick loop -> shutdown
	// -> Engine::Close. shutdown runs exactly once after init is entered, including
	// when init fails, and runs while the engine remains live. Engine-init failure
	// closes partial engine state without retaining an app engine or entering init
	// or shutdown. Setup's
	// nonzero result is preserved; init, tick, or Close failure returns 1.
	int main(int argc, char** argv);

protected:
	// valid only during init, tick, and shutdown. Setup deliberately runs before
	// device creation so it can finalize engineConfig_ or exit without vulkan work.
	[[nodiscard]] oa::Engine& engine() noexcept {
		OA_ASSERT(engineOwner_ != nullptr);
		return *engineOwner_;
	}
	[[nodiscard]] const oa::Engine& engine() const noexcept {
		OA_ASSERT(engineOwner_ != nullptr);
		return *engineOwner_;
	}
	[[nodiscard]] bool hasEngine() const noexcept { return engineOwner_ != nullptr; }

	virtual int setup(int argc, char** argv) = 0;
	virtual oa::Status init() { return oa::Status::ok(); }
	virtual oa::Status tick() = 0;
	// release derived GPU-owning members here. main closes the engine after this
	// callback and before the derived application object is destroyed.
	virtual void shutdown() {}

	oa::EngineConfig engineConfig_;

private:
	oa::UniquePtr<oa::Engine> engineOwner_;
};

} // namespace oa

// Beginner application spelling. OA_MAIN emits the process entry point and a
// private body function with one lexical Engine owner behind it. The body can
// use the borrowed `engine` and the ordinary `argc` / `argv` names directly.
// Every return from the body still passes through oa::withEngine, preserving
// explicit close and exit-status handling. The short application profile loads
// only the shader variants required by each executable graph; callers that need
// eager preload use the EngineConfig overload directly.
#define OA_DETAIL_MAIN_IMPL(inAppName, inPresentationMode, inLine) \
	static int OA_PP_GLUE(oaMainBody, inLine)( \
		[[maybe_unused]] oa::Engine& engine, \
		[[maybe_unused]] int argc, \
		[[maybe_unused]] char** argv); \
	int main(int argc, char** argv) { \
		return oa::withEngine( \
			(inAppName), \
			[argc, argv](oa::Engine& engine) { \
				return OA_PP_GLUE(oaMainBody, inLine)(engine, argc, argv); \
			}, \
			(inPresentationMode)); \
	} \
	static int OA_PP_GLUE(oaMainBody, inLine)( \
		[[maybe_unused]] oa::Engine& engine, \
		[[maybe_unused]] int argc, \
		[[maybe_unused]] char** argv)

#define OA_DETAIL_MAIN(inAppName, inPresentationMode, inLine) \
	OA_DETAIL_MAIN_IMPL(inAppName, inPresentationMode, inLine)

#define OA_MAIN_MODE(inAppName, inPresentationMode) \
	OA_DETAIL_MAIN(inAppName, inPresentationMode, __LINE__)

#define OA_MAIN(inAppName) \
	OA_MAIN_MODE(inAppName, oa::PresentationMode::None)

#define OA_MAIN_PREVIEW(inAppName) \
	OA_MAIN_MODE( \
		inAppName, \
		argc > 1 and oa::StringView(argv[1]) == "--preview" \
			? oa::PresentationMode::Swapchain \
			: oa::PresentationMode::None)
