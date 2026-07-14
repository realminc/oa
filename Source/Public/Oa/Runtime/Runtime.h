// OaRuntime — High-level fixture/wrapper for simplified ML and compute applications
//
// **Purpose**: Eliminate boilerplate engine/context initialization for tutorials,
// ML training scripts, and compute applications. Provides a Pythonic API similar
// to the Python bindings but for C++.
//
// **Usage**:
//   // Simplest form - auto-init with defaults
//   OaRuntime rt;
//   auto x = OaFnMatrix::Add(a, b);  // Uses default context
//
//   // With config
//   OaRuntime rt(OaRuntimeConfig{.Precision = OaPrecision::BF16});
//
//   // Explicit control
//   OaRuntime rt;
//   rt.Execute();  // Manual execution
//   rt.Sync();
//
//   // RAII scope for automatic execution
//   {
//       OaRuntime::Scope scope;
//       auto y = OaFnMatrix::Mul(x, w);
//   }  // Auto Execute() + Sync()
//
// **Design Philosophy**:
// - Zero-config initialization for 90% of use cases
// - Automatic engine + context creation and cleanup
// - RAII-based lifetime management
// - Compatible with existing OaContext/OaEngine APIs
// - No global state pollution (uses existing OaContext::GetDefault())

#pragma once

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/EnvFlag.h>

// ═════════════════════════════════════════════════════════════════════════════
// Configuration
// ═════════════════════════════════════════════════════════════════════════════

struct OaRuntimeConfig {
	// Device selection
	OaDevicePreference DevicePref = OaDevicePreference::Discrete;
	OaI32 DeviceIndex = -1;  // -1 = auto, 0+ = specific device

	// Precision
	OaPrecision Precision = OaPrecision::FP32;
	OaNumericMode NumericMode = OaNumericMode::Fast;

	// Presentation (for graphics apps)
	OaPresentationMode PresentationMode = OaPresentationMode::None;

	// Validation
	OaBool EnableValidation = false;

	// Application info
	OaString AppName = "OaApp";

	// Auto-execute on scope exit (default true for convenience)
	OaBool AutoExecute = true;

	// Register as global (allows OaContext::GetDefault() to work)
	OaBool RegisterAsGlobal = true;
};

// ═════════════════════════════════════════════════════════════════════════════
// OaRuntime — High-level engine + context wrapper
// ═════════════════════════════════════════════════════════════════════════════

class OaRuntime {
public:
	// ─── Construction ────────────────────────────────────────────────────────

	// Default constructor - auto-init with sensible defaults
	OaRuntime() : OaRuntime(OaRuntimeConfig{}) {}

	// Construct with custom config
	explicit OaRuntime(const OaRuntimeConfig& InConfig) {
		Config_ = InConfig;
		
		// Apply device index from config or environment
		if (Config_.DeviceIndex >= 0) {
			OaString idxStr = OaString(std::to_string(Config_.DeviceIndex).c_str());
#if defined(_WIN32)
			_putenv_s("OA_DEVICE", idxStr.c_str());
#else
			::setenv("OA_DEVICE", idxStr.c_str(), 1);
#endif
		}

		// Build engine config
		OaEngineConfig ecfg;
		ecfg.DevicePref = Config_.DevicePref;
		ecfg.Precision = Config_.Precision;
		ecfg.NumericMode = Config_.NumericMode;
		ecfg.PresentationMode = Config_.PresentationMode;
		ecfg.EnableValidation = Config_.EnableValidation;
		ecfg.AppName = Config_.AppName;
		ecfg.RegisterAsGlobal = Config_.RegisterAsGlobal;
		
		if (Config_.DeviceIndex >= 0) {
			ecfg.DevicePref = OaDevicePreference::ByIndex;
			ecfg.DeviceIndex = static_cast<OaU32>(Config_.DeviceIndex);
		}

		// Create engine
		auto result = OaComputeEngine::Create(ecfg);
		if (!result) {
			OA_LOG_ERROR(OaLogComponent::Core, "OaRuntime: Failed to create engine: %s",
				result.GetStatus().ToString().c_str());
			return;
		}

		// Take ownership of the pinned, heap-allocated engine (stable address).
		// OaRuntime keeps a raw owning pointer and deletes it in its destructor.
		Engine_ = result.GetValue().Release();
		
		// Create context
		Context_ = OaContext::Create(Engine_);
		if (!Context_) {
			OA_LOG_ERROR(OaLogComponent::Core, "OaRuntime: Failed to create context");
			Engine_->Destroy();
			delete Engine_;
			Engine_ = nullptr;
			return;
		}

		Valid_ = true;
	}

	// ─── Destruction ─────────────────────────────────────────────────────────

	~OaRuntime() {
		if (Valid_ && Config_.AutoExecute) {
			// Auto-execute and sync on destruction
			(void)Execute();
			(void)Sync();
		}

		if (Context_) {
			// Clear context before engine destruction
			Context_->Clear();
			OaContext::SetDefault(nullptr);
			delete Context_;
			Context_ = nullptr;
		}

		if (Engine_) {
			Engine_->Destroy();
			delete Engine_;
			Engine_ = nullptr;
		}
	}

	// Non-copyable, non-movable (owns engine/context)
	OaRuntime(const OaRuntime&) = delete;
	OaRuntime& operator=(const OaRuntime&) = delete;
	OaRuntime(OaRuntime&&) = delete;
	OaRuntime& operator=(OaRuntime&&) = delete;

	// ─── Execution ───────────────────────────────────────────────────────────

	// Execute recorded operations
	[[nodiscard]] OaStatus Execute() {
		if (!Valid_ || !Context_) return OaStatus::Error("Runtime not initialized");
		return Context_->Execute();
	}

	// Synchronize with GPU
	[[nodiscard]] OaStatus Sync() {
		if (!Valid_ || !Context_) return OaStatus::Error("Runtime not initialized");
		return Context_->Sync();
	}

	// Clear recorded operations
	void Clear() {
		if (Valid_ && Context_) {
			Context_->Clear();
		}
	}

	// ─── Accessors ───────────────────────────────────────────────────────────

	[[nodiscard]] bool IsValid() const noexcept { return Valid_; }
	[[nodiscard]] OaContext& GetContext() const { return *Context_; }
	[[nodiscard]] OaComputeEngine& GetEngine() const { return *Engine_; }
	[[nodiscard]] OaVkDevice& GetDevice() const { return Engine_->Device; }

	// Convenience accessors
	[[nodiscard]] OaPrecision GetPrecision() const { return Engine_->GetPrecision(); }
	[[nodiscard]] OaStringView GetDeviceName() const { return Engine_->DeviceName(); }

	// ─── RAII Scope Helper ───────────────────────────────────────────────────

	// Automatic execution and sync at scope exit
	// Use this for isolated compute blocks
	class Scope {
	public:
		Scope() : Runtime_(new OaRuntime()) {}
		
		explicit Scope(const OaRuntimeConfig& InConfig) 
			: Runtime_(new OaRuntime(InConfig)) {}

		~Scope() {
			if (Runtime_) {
				delete Runtime_;
			}
		}

		Scope(const Scope&) = delete;
		Scope& operator=(const Scope&) = delete;
		Scope(Scope&&) = delete;
		Scope& operator=(Scope&&) = delete;

		[[nodiscard]] OaRuntime& Get() const { return *Runtime_; }
		[[nodiscard]] OaContext& GetContext() const { return Runtime_->GetContext(); }

	private:
		OaRuntime* Runtime_;
	};

private:
	OaRuntimeConfig Config_;
	OaComputeEngine* Engine_ = nullptr;
	OaContext* Context_ = nullptr;
	bool Valid_ = false;
};

// ═════════════════════════════════════════════════════════════════════════════
// Convenience Macros
// ═════════════════════════════════════════════════════════════════════════════

// Initialize runtime with default config
// Usage: OA_INIT_RUNTIME();
#define OA_INIT_RUNTIME() \
	OaRuntime __oa_runtime_guard

// Initialize runtime with custom config
// Usage: OA_INIT_RUNTIME_CFG(OaRuntimeConfig{.Precision = OaPrecision::BF16});
#define OA_INIT_RUNTIME_CFG(config) \
	OaRuntime __oa_runtime_guard(config)

// Initialize runtime with precision shorthand
// Usage: OA_INIT_RUNTIME_BF16();
#define OA_INIT_RUNTIME_BF16() \
	OaRuntime __oa_runtime_guard(OaRuntimeConfig{.Precision = OaPrecision::BF16})

#define OA_INIT_RUNTIME_FP32() \
	OaRuntime __oa_runtime_guard(OaRuntimeConfig{.Precision = OaPrecision::FP32})

#define OA_INIT_RUNTIME_FP16() \
	OaRuntime __oa_runtime_guard(OaRuntimeConfig{.Precision = OaPrecision::FP16})

// Execute and sync manually (disables auto-execute)
// Usage: OA_INIT_RUNTIME_MANUAL(); ... OA_EXECUTE(); OA_SYNC();
#define OA_INIT_RUNTIME_MANUAL() \
	OaRuntime __oa_runtime_guard(OaRuntimeConfig{.AutoExecute = false})

#define OA_EXECUTE() \
	__oa_runtime_guard.Execute()

#define OA_SYNC() \
	__oa_runtime_guard.Sync()

#define OA_RUNTIME() \
	__oa_runtime_guard

// ═════════════════════════════════════════════════════════════════════════════
// Python-style initialization function
// ═════════════════════════════════════════════════════════════════════════════

// Global runtime instance for Python-style usage
// Note: This is a convenience for simple scripts. Production code should
// prefer explicit OaRuntime instances for better lifetime control.
namespace OaRuntimeGlobal {
	inline OaRuntime* GlobalRuntime = nullptr;

	// Initialize global runtime (Python-style)
	// Returns true on success
	inline bool Init(const OaRuntimeConfig& InConfig = {}) {
		if (GlobalRuntime) {
			OA_LOG_WARN(OaLogComponent::Core, "OaRuntimeGlobal::Init called when runtime already exists");
			return true;
		}
		GlobalRuntime = new OaRuntime(InConfig);
		return GlobalRuntime->IsValid();
	}

	// Shutdown global runtime
	inline void Shutdown() {
		if (GlobalRuntime) {
			delete GlobalRuntime;
			GlobalRuntime = nullptr;
		}
	}

	// Get global runtime (must call Init first)
	inline OaRuntime& Get() {
		if (!GlobalRuntime) {
			OA_LOG_ERROR(OaLogComponent::Core, "OaRuntimeGlobal::Get called before Init");
			static OaRuntime dummy;
			return dummy;
		}
		return *GlobalRuntime;
	}

	// Check if initialized
	inline bool IsInitialized() {
		return GlobalRuntime != nullptr && GlobalRuntime->IsValid();
	}
}

// Python-style init function (matches Python bindings)
inline bool OaInitComputeEngine(const OaRuntimeConfig& InConfig = {}) {
	return OaRuntimeGlobal::Init(InConfig);
}

inline void OaShutdownComputeEngine() {
	OaRuntimeGlobal::Shutdown();
}

