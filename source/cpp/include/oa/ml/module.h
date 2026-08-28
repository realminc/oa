// oa::Module — slim NN base class.
//
// One class. layers override forward(oa::Matrix) -> oa::Matrix. Composite modules
// declare structure via registerParameter / registerModule in the constructor;
// everything downstream (parameter walk, optimizer binding, persistence) is
// derived from that registration.
//
// Mode (train/eval) is ML module state and propagates through registered child
// modules. The runtime recorder owns execution, never model policy. backward
// passes are hand-wired via oa:: free functions until implicit autograd covers
// the remaining manual paths. save/load are generic tree-walk helpers.

#pragma once

#include <oa/core/matrix.h>
#include <oa/core/fnMatrix.h>  // Transitive: many ML headers expect FnMatrix via module.h

namespace oa {

class ModelFile;
class Optimizer;
class Engine;

// oa::Parameter — named trainable tensor with live gradient access.
class Parameter {
public:
	oa::String name;
	Matrix   data;
	bool     requiresGrad = true;

	// Gradient — SINGLE SOURCE OF TRUTH. The grad buffer is owned by data's autograd
	// meta (allocated lazily by setRequiresGrad / accumulateGrad). grad() resolves to
	// that live buffer every call, so it can never be a stale snapshot.
	//
	//   - non-const grad() → mutable lvalue ref to the live grad (fill / assign / &).
	//   - const grad()     → by-value handle sharing the live buffer (read-only).
	[[nodiscard]] Matrix& grad() { return data.mutGradMatrix(); }
	[[nodiscard]] Matrix  grad() const { return data.gradMatrix(); }
};

/// Dotted module path + parameter pointer (e.g. "blocks.0.gate.weight").
struct NamedParameter {
	oa::String    path;
	Parameter*  param = nullptr;
};

// oa::ModuleBuffer — persistent/non-trainable buffer registered on a module.
class ModuleBuffer {
public:
	oa::String name;
	Matrix   data;
	bool     persistent = true;
};

// oa::Module — base class for all neural network modules.
class Module {
public:
	virtual ~Module() = default;

	Module(const Module&) = delete;
	Module& operator=(const Module&) = delete;

	// forward — the only virtual method layers override.
	virtual Matrix forward(const Matrix& inInput);
	Matrix operator()(const Matrix& inInput) { return forward(inInput); }

	// training mode — propagates to every registered child.
	// eval() is equivalent to train(false).
	// Newly registered children inherit the parent's current mode.
	void train(oa::Bool inTraining = true);
	void eval() { train(false); }
	[[nodiscard]] oa::Bool isTraining() const noexcept { return training_; }

	class ScopedEval {
	public:
		explicit ScopedEval(Module& inModule)
			: module_(inModule)
			, previous_(inModule.isTraining())
		{
			inModule.eval();
		}
		~ScopedEval() { module_.train(previous_); }
		ScopedEval(const ScopedEval&) = delete;
		ScopedEval& operator=(const ScopedEval&) = delete;
		ScopedEval(ScopedEval&&) noexcept = delete;
		ScopedEval& operator=(ScopedEval&&) noexcept = delete;

	private:
		Module& module_;
		oa::Bool  previous_;
	};

	// Parameter management
	//
	// WARNING: parameters() returns ONLY direct parameters (non-recursive), unlike
	// PyTorch's .parameters(). For nested modules use allParameterPtrs() instead.
	// FOOTGUN: module->parameters()[0] on a nested module is OOB. Always check
	// .size() first or use allParameterPtrs().
	[[nodiscard]] oa::Vector<Parameter>& parameters() { return params_; }
	[[nodiscard]] const oa::Vector<Parameter>& parameters() const { return params_; }

	// Recursive parameter collection (use these for nested modules)
	[[nodiscard]] oa::Vector<Parameter>        allParameters();
	[[nodiscard]] oa::Vector<Parameter*>       allParameterPtrs();
	[[nodiscard]] oa::Vector<NamedParameter>   allNamedParameterPtrs();
	[[nodiscard]] virtual oa::I64           numParameters() const;
	void registerModule(oa::StringView inName, oa::SharedPtr<Module> inModule);
	void registerParameter(oa::StringView inName, Matrix inData, bool inRequiresGrad = true);

	// persistent/non-trainable state. buffers never receive gradients and never
	// participate in optimizer parameter traversal or numParameters().
	[[nodiscard]] oa::Vector<ModuleBuffer>& buffers() { return buffers_; }
	[[nodiscard]] const oa::Vector<ModuleBuffer>& buffers() const { return buffers_; }
	[[nodiscard]] oa::Vector<ModuleBuffer*> allBufferPtrs(bool inPersistentOnly = false);
	void registerBuffer(oa::StringView inName, Matrix inData, bool inPersistent = true);

	// Sub-modules
	class NamedChild {
	public:
		oa::String            name;
		oa::SharedPtr<Module> module;
	};
	[[nodiscard]] oa::Vector<NamedChild>& children() { return children_; }
	[[nodiscard]] const oa::Vector<NamedChild>& children() const { return children_; }

	// Persistence — non-virtual generic tree walks. Builds dotted parameter paths
	// from registerModule/registerParameter names (e.g. "fc1.weight"). arch-specific
	// loaders (SafeTensors, sharded LLM weights) live as separate helper fns that
	// write into an already-constructed module.
	//
	// Optimizer overloads bundle state into the same .oam file so resume
	// training is one call per side.
	[[nodiscard]] oa::Status save(Engine& inEngine, const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(Engine& inEngine, const oa::String& inPath);
	[[nodiscard]] oa::Status save(Engine& inEngine, const oa::String& inPath,	const Optimizer& inOptimizer) const;
	[[nodiscard]] oa::Status load(Engine& inEngine, const oa::String& inPath,	Optimizer& inOptimizer);

	// Lower-level: dump self/restore self into a caller-owned ModelFile (no file I/O).
	[[nodiscard]] oa::Status saveTo(Engine& inEngine, ModelFile& outFile) const;
	// validate the complete module tree before mutating live state.
	[[nodiscard]] oa::Status loadFrom(Engine& inEngine, const ModelFile& inFile);

	// Info
	void setName(oa::StringView inName) { name_ = oa::String(inName); }
	[[nodiscard]] const oa::String& getName() const { return name_; }
	[[nodiscard]] oa::Device        getDevice() const { return device_; }

protected:
	Module() = default;

	// save/load tree walk helpers.
	[[nodiscard]] oa::Status saveWalk(Engine& inEngine, ModelFile& outFile, const oa::String& inPrefix) const;
	[[nodiscard]] oa::Status validateLoadWalk(const ModelFile& inFile, const oa::String& inPrefix) const;
	[[nodiscard]] oa::Status loadWalk(Engine& inEngine, const ModelFile& inFile, const oa::String& inPrefix);

	oa::String            name_;
	oa::Vector<Parameter>    params_;
	oa::Vector<ModuleBuffer> buffers_;
	oa::Vector<NamedChild>   children_;
	oa::Bool              training_ = true;
	oa::Device            device_{oa::DeviceType::VkDiscrete, 0};
};

} // namespace oa
