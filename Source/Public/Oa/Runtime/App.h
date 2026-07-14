// OaApp — minimal base (currently unused; reserved for future CLI tools).
// OaComputeApp — compute apps with tick loop: OaComputeEngine + Setup/Init/Tick/Shutdown.
// OaGraphicsApp — full-featured windowed app with graphics/GUI: window + input + compute engine.
//
// Usage:
//   struct MyApp : OaComputeApp {
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

// Vulkan application — wraps OaComputeEngine create/destroy around the tick loop.
// Subclass for training, inference, compute apps.
// Device selection: EngineConfig_.DevicePref / DeviceIndex / MeshVulkanIndices (see OaEngineConfig).
class OaComputeApp : public OaApp {
public:
	bool IsRunning = true;

	// Pinned engine ownership. RtStorage_ holds the (heap, stable-address) engine;
	// it is default-constructed empty here — no Vulkan work — and initialized in
	// place by Main() once Setup() has filled EngineConfig_. Rt is a stable
	// reference to it so all app code reads `Rt.Method()` / passes `Rt` unchanged.
	OaUniquePtr<OaComputeEngine> RtStorage_ = OaMakeUniquePtr<OaComputeEngine>();
	OaComputeEngine&             Rt         = *RtStorage_;

	// Register an extension before Main() — populates adapters and kernel table.
	// Extension pointer must outlive the engine (use static singletons).
	void AddExtension(OaExtension* InExtension) { Extensions_.PushBack(InExtension); }

	// Orchestrates: Setup -> Engine::Create (with extensions) -> Init -> tick loop -> Shutdown
	int Main(int argc, char** argv);

protected:
	virtual OaStatus Init() { return OaStatus::Ok(); }
	virtual OaStatus Tick() = 0;
	virtual void Shutdown() {}

	OaEngineConfig    EngineConfig_;
	OaVec<OaExtension*> Extensions_;
};

class OaGraphicsApp : public OaComputeApp {};
