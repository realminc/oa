// OaApp — minimal base (currently unused; reserved for future CLI tools).
// OaComputeApp — compute apps with tick loop: OaEngine + Setup/Init/Tick/Shutdown.
// Windowed apps compose OaPresenter with the same engine; there is no graphics
// application subclass mirroring the removed engine hierarchy.
//
// Usage:
//   class MyApp : public OaComputeApp {
//   public:
//       int Setup(int argc, char** argv) override { /* parse args */ return 0; }
//       OaStatus Tick() override { /* do work */ IsRunning = false; return OaStatus::Ok(); }
//   };
//   int main(int argc, char** argv) { MyApp app; return app.Main(argc, argv); }

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Extension.h>
#include <Oa/Runtime/Engine.h>

// Base application — no Vulkan dependency.
// Minimal base for future CLI tools or non-GPU apps.
// Currently not used directly; all apps inherit from OaComputeApp.
class OaApp {
public:
	virtual ~OaApp() = default;

protected:
	virtual int Setup(int argc, char** argv) = 0;
};

// Vulkan application — owns one pinned OaEngine around the tick loop.
// Subclass for training, inference, compute apps.
// Device selection: EngineConfig_.DevicePref / DeviceIndex / MeshVulkanIndices (see OaEngineConfig).
class OaComputeApp : public OaApp {
public:
	bool IsRunning = true;

	// Pinned engine ownership. RtStorage_ holds the (heap, stable-address) engine;
	// it is default-constructed empty here — no Vulkan work — and initialized in
	// place by Main() once Setup() has filled EngineConfig_. Rt is a stable
	// reference to it so all app code reads `Rt.Method()` / passes `Rt` unchanged.
	OaUniquePtr<OaEngine> RtStorage_ = OaMakeUniquePtr<OaEngine>();
	OaEngine&             Rt         = *RtStorage_;

	// Register an extension before Main() — populates adapters and kernel table.
	// Extension pointer must outlive the engine (use static singletons).
	void AddExtension(OaExtension* InExtension) { Extensions_.PushBack(InExtension); }

	// Orchestrates: Setup -> Engine::InitInPlace -> Init -> tick loop -> Shutdown
	// -> Engine::Close. Shutdown runs exactly once after Init is entered, including
	// when Init fails, and runs while the engine remains live. Engine-init failure
	// closes partial engine state without entering Init or Shutdown. Setup's
	// nonzero result is preserved; Init, Tick, or Close failure returns 1.
	int Main(int argc, char** argv);

protected:
	virtual OaStatus Init() { return OaStatus::Ok(); }
	virtual OaStatus Tick() = 0;
	// Release derived GPU-owning members here. Main closes the engine after this
	// callback and before the derived application object is destroyed.
	virtual void Shutdown() {}

	OaEngineConfig    EngineConfig_;
	OaVec<OaExtension*> Extensions_;
};
