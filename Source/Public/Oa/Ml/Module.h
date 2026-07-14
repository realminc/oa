// OaModule — slim NN base (OaModule.md Phase 2)
//
// One class. Layers override Forward(OaMatrix) -> OaMatrix. Composite modules
// declare structure via RegisterParameter / RegisterModule in the constructor;
// everything downstream (parameter walk, optimizer binding, persistence) is
// derived from that registration.
//
// Mode (train/eval) lives on OaContext (see OaContext::ScopedEval). Backward
// passes are hand-wired via OaFnMatrix::*Bwd until Phase 3 implicit autograd
// lands. Save/Load are tree-walk helpers (Phase 2 follow-up, Task #6).
//
// See: Docs/Rewrite/ThisIsTheKey/OaModule.md

#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/FnMatrix.h>  // Transitive: many ML headers expect OaFnMatrix via Module.h

class OamModel;
class OaOptimizer;

// PARAMETER

class OaParameter {
public:
	OaString Name;
	OaMatrix Data;
	bool RequiresGrad = true;

	// Gradient — SINGLE SOURCE OF TRUTH. The grad buffer is owned by Data's autograd
	// meta (allocated lazily by SetRequiresGrad / AccumulateGrad). Grad() resolves to
	// that live buffer every call, so it can never be a stale snapshot. The old
	// `OaMatrix Grad` field was a one-time copy taken at register time; it silently
	// desynced whenever a param's Data was reassigned/reshaped (forcing manual
	// `.Grad = .GradMatrix()` re-syncs across the NN modules) and was the long-hunted
	// optimizer-divergence footgun.
	//
	//   - non-const Grad() → mutable lvalue ref to the live grad (Fill / assign / &).
	//   - const Grad()     → by-value handle sharing the live buffer (read-only).
	[[nodiscard]] OaMatrix& Grad() { return Data.MutGradMatrix(); }
	[[nodiscard]] OaMatrix  Grad() const { return Data.GradMatrix(); }
};

/// Dotted module path + parameter pointer (e.g. "blocks.0.gate.weight").
struct OaNamedParameter {
	OaString Path;
	OaParameter* Param = nullptr;
};

class OaModuleBuffer {
public:
	OaString Name;
	OaMatrix Data;
	bool Persistent = true;
};

// MODULE

class OaModule {
public:
	virtual ~OaModule() = default;

	OaModule(const OaModule&) = delete;
	OaModule& operator=(const OaModule&) = delete;

	// Forward — the only virtual layers override.
	virtual OaMatrix Forward(const OaMatrix& InInput);
	OaMatrix operator()(const OaMatrix& InInput) { return Forward(InInput); }

	// Parameter management
	//
	// WARNING: Parameters() returns ONLY direct parameters (non-recursive), unlike
	// PyTorch's .parameters(). For nested modules (e.g., GRU with internal Linear layers),
	// this returns an EMPTY vector. Use AllParameterPtrs() for recursive collection.
	//
	// FOOTGUN: module->Parameters()[0] on a nested module is OOB (SIGSEGV in Release).
	// Always check .Size() first or use AllParameterPtrs() instead.
	[[nodiscard]] OaVec<OaParameter>& Parameters() { return Params_; }
	[[nodiscard]] const OaVec<OaParameter>& Parameters() const { return Params_; }
	
	// Recursive parameter collection (use these for nested modules)
	[[nodiscard]] OaVec<OaParameter> AllParameters();
	[[nodiscard]] OaVec<OaParameter*> AllParameterPtrs();
	[[nodiscard]] OaVec<OaNamedParameter> AllNamedParameterPtrs();
	[[nodiscard]] virtual OaI64 NumParameters() const;
	void RegisterModule(OaStringView InName, OaSharedPtr<OaModule> InModule);
	void RegisterParameter(OaStringView InName, OaMatrix InData, bool InRequiresGrad = true);

	// Persistent/non-trainable state. Buffers never receive gradients and never
	// participate in optimizer parameter traversal or NumParameters().
	[[nodiscard]] OaVec<OaModuleBuffer>& Buffers() { return Buffers_; }
	[[nodiscard]] const OaVec<OaModuleBuffer>& Buffers() const { return Buffers_; }
	[[nodiscard]] OaVec<OaModuleBuffer*> AllBufferPtrs(bool InPersistentOnly = false);
	void RegisterBuffer(OaStringView InName, OaMatrix InData, bool InPersistent = true);

	// Sub-modules
	class NamedChild {
	public:
		OaString Name;
		OaSharedPtr<OaModule> Module;
	};
	[[nodiscard]] OaVec<NamedChild>& Children() { return Children_; }
	[[nodiscard]] const OaVec<NamedChild>& Children() const { return Children_; }

	// Persistence — non-virtual generic tree walks. Builds dotted parameter paths
	// from RegisterModule/RegisterParameter names (e.g. "fc1.weight"). Arch-specific
	// loaders (SafeTensors, sharded LLM weights) live as separate helper fns that
	// write into an already-constructed module.
	//
	// Two-arg overloads bundle optimizer state into the same .oam file so resume
	// training is one call per side. The OamModel format already has slots for it
	// (OamOptimizerHeader + AdamM/AdamV), so combined checkpoints round-trip cleanly.
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);
	[[nodiscard]] OaStatus Save(const OaString& InPath, const OaOptimizer& InOptimizer) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath, OaOptimizer& InOptimizer);

	// Lower-level: dump self/restore self into a caller-owned OamModel (no file I/O).
	// Useful when composing multiple modules + optimizer + custom metadata.
	void SaveTo(OamModel& OutOam) const;
	void LoadFrom(const OamModel& InOam);

	// Info
	void SetName(OaStringView InName) { Name_ = OaString(InName); }
	[[nodiscard]] const OaString& GetName() const { return Name_; }
	[[nodiscard]] OaDevice GetDevice() const { return Device_; }

protected:
	OaModule() = default;

	// Save/Load tree walk: emit (dotted-path, param) entries for self then children.
	void SaveWalk(class OamModel& OutOam, const OaString& InPrefix) const;
	void LoadWalk(const class OamModel& InOam, const OaString& InPrefix);

	OaString Name_;
	OaVec<OaParameter> Params_;
	OaVec<OaModuleBuffer> Buffers_;
	OaVec<NamedChild> Children_;
	OaDevice Device_{OaDeviceType::VkDiscrete, 0};
};
