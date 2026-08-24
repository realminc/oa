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
		assert(engineOwner_ != nullptr);
		return *engineOwner_;
	}
	[[nodiscard]] const oa::Engine& engine() const noexcept {
		assert(engineOwner_ != nullptr);
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
