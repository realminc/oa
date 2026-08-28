#include <oa/ml/optim.h>
#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/cString.h>
#include <oa/core/validation.h>

static oa::Matrix getParamGrad(oa::Parameter* inP) {
	return inP->grad();  // live grad (single source of truth: Data's autograd meta)
}

void oa::Sgd::step() {
	++step_;
	auto& ctx = oa::ExecutionSession::getActive();
	// fp32 master weights for any low-precision (bf16) params — the "Sgd" kernel
	// then runs on the fp32 master (grad upcast) and the bf16 weight is re-derived.
	prepareMasters();

	for (oa::Usize i = 0; i < params_.size(); ++i) {
		auto* p = params_[i];
		oa::Matrix grad = getParamGrad(p);
		if (!grad.hasStorage()) continue;

		oa::Matrix gradUse = masterGrad(i, grad);
		oa::Matrix& weight = masterOrData(i);
		oa::U32 n = static_cast<oa::U32>(weight.numElements());

		struct { oa::U32 Count; oa::F32 lr; oa::F32 weightDecay; }
			push{n, lr_, weightDecay_};
		oa::BufferAccess access[] = {oa::BufferAccess::ReadWrite, oa::BufferAccess::Read};
		ctx.add( "Sgd", {&weight, &gradUse}, access, &push, sizeof(push), oa::divCeil(n, 256));
		writebackMaster(i);
	}
}

void oa::Sgd::zeroGrad() {
	// Zero the single source of truth (Data's autograd grad) GPU-side. oa::Matrix::zeroGrad
	// self-guards (no-op if the param has no grad buffer) and records a Fill kernel — never
	// a host memset, which was a silent no-op on GPU buffers that let accumulateGrad grow
	// the grad without bound across steps → divergence.
	for (auto* p : params_) { p->data.zeroGrad(); }
}

// SGD has no per-param momentum buffer in the current impl (the kernel ignores
// momentum_). Persistence is header-only: lr, weightDecay, step, plus momentum
// stashed in beta1 (repurposed) so future SgdMomentum work can recover it.

oa::Status oa::Sgd::saveTo(oa::Engine& inEngine, ModelFile& outFile) const {
	(void)inEngine;
	outFile.optimizerPresent = true;
	outFile.optimizer = ModelOptimizerState{};
	constexpr char kType[] = "SGD";
	oa::memcpy(outFile.optimizer.type, kType, sizeof(kType));
	outFile.optimizer.lr          = lr_;
	outFile.optimizer.beta1       = momentum_;  // repurposed: SGD has no beta1
	outFile.optimizer.weightDecay = weightDecay_;
	outFile.optimizer.step        = static_cast<oa::I64>(step_);
	outFile.optimizer.numParams   = 0;
	outFile.adamM.clear();
	outFile.adamV.clear();
	return oa::Status::ok();
}

oa::Status oa::Sgd::validateLoad(const ModelFile& inFile) const {
	if (not inFile.hasOptimizer()
		or oa::strncmp(inFile.optimizer.type, "SGD", sizeof(inFile.optimizer.type)) != 0)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"SGD checkpoint optimizer state is missing or has the wrong type");
	}
	if (not inFile.adamM.empty() or not inFile.adamV.empty()) {
		return oa::Status::error(oa::StatusCode::CheckpointCorrupt,
			"SGD checkpoint contains unexpected optimizer payloads");
	}
	return oa::Status::ok();
}

oa::Status oa::Sgd::loadFrom(oa::Engine& inEngine, const ModelFile& inFile) {
	(void)inEngine;
	OA_RETURN_IF_ERROR(validateLoad(inFile));
	lr_          = inFile.optimizer.lr;
	momentum_    = inFile.optimizer.beta1;
	weightDecay_ = inFile.optimizer.weightDecay;
	step_        = static_cast<oa::U64>(inFile.optimizer.step);
	resetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights
	return oa::Status::ok();
}
