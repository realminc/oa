#include <oa/ml/optim.h>
#include <oa/ml/fnOptim.h>
#include <oa/ml/modelFile.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/log.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>

#include <cstring>

static oa::Matrix getParamGrad(oa::Parameter* inP) {
	return inP->grad();  // live grad (single source of truth: Data's autograd meta)
}

static void ensureMomentumBuffers(oa::Vec<oa::Parameter*>& inParams, oa::Vec<oa::Matrix>& inOutMomentum) {
	if (not inOutMomentum.empty()) return;
	for (auto* p : inParams) {
		inOutMomentum.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
	}
}

void oa::Muon::step() {
	++step_;

	// Lazy-initialize momentum buffers on first step
	ensureMomentumBuffers(params_, momentum_);
	// fp32 master weights for any low-precision (bf16) params — the Muon NS5/apply
	// runs on the fp32 master (grad upcast) and the bf16 weight is re-derived.
	prepareMasters();

	// apply Muon update to each parameter using oa::FnOptim stateless function
	for (oa::Usize i = 0; i < params_.size(); ++i) {
		auto* p = params_[i];
		oa::Matrix grad = getParamGrad(p);
		if (!grad.hasStorage()) continue;

		oa::Matrix gradUse = masterGrad(i, grad);
		oa::FnOptim::muonStep(
			masterOrData(i),   // inOutParam: fp32 master (bf16) or Data (fp32)
			momentum_[i],      // inOutMomentum (mutated)
			gradUse,           // inGrad: fp32 upcast (bf16) or grad (fp32)
			lr_, beta_, weightDecay_, eps_, ns5Iterations_
		);
		writebackMaster(i);
	}
}

void oa::Muon::zeroGrad() {
	// Zero the single source of truth GPU-side (self-guarded Fill kernel, never a host
	// memset). See oa::Sgd::zeroGrad.
	for (auto* p : params_) { p->data.zeroGrad(); }
}

// ─── Persistence ──────────────────────────────────────────────────────────

oa::Status oa::Muon::saveTo(oa::Engine& inEngine, ModelFile& outFile) const {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	// header (hyperparams + step count). numParams is total flat element count.
	outFile.optimizerPresent = true;
	outFile.optimizer = ModelOptimizerState{};
	std::strncpy(outFile.optimizer.type, "Muon", sizeof(outFile.optimizer.type) - 1);
	outFile.optimizer.lr = lr_;
	outFile.optimizer.beta1 = beta_;  // Repurposed: Muon has no beta1/beta2, store Beta in beta1
	outFile.optimizer.beta2 = 0.0f;   // Not used by Muon
	outFile.optimizer.eps = eps_;
	outFile.optimizer.weightDecay = weightDecay_;
	outFile.optimizer.step = static_cast<oa::I64>(step_);

	// momentum_ may not be allocated yet (no step() called). then nothing to save.
	if (momentum_.empty()) {
		outFile.optimizer.numParams = 0;
		outFile.adamM.clear();
		outFile.adamV.clear();
		return oa::Status::ok();
	}

	// drain pending GPU writes to momentum_ so the memcpy sees the latest state.
	OA_RETURN_IF_ERROR(context.submitAndWait());

	oa::I64 total = 0;
	for (const auto& m : momentum_) total += m.numElements();
	outFile.optimizer.numParams = static_cast<oa::U64>(total);
	outFile.adamM.resize(total);
	outFile.adamV.clear();  // Muon has no second moment, so adamV is unused

	oa::I64 off = 0;
	for (oa::Usize i = 0; i < momentum_.size(); ++i) {
		oa::I64 n = momentum_[i].numElements();
		const auto bytes = static_cast<oa::U64>(n) * sizeof(oa::F32);
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(momentum_[i], outFile.adamM.data() + off, bytes));
		off += n;
	}
	return oa::Status::ok();
}

oa::Status oa::Muon::validateLoad(const ModelFile& inFile) const {
	if (not inFile.hasOptimizer() or not modelFileIsMuonOnly(inFile.optimizer)) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Muon checkpoint optimizer state is missing or has the wrong type");
	}
	oa::U64 expected = 0;
	for (const auto* parameter : params_) {
		expected += static_cast<oa::U64>(parameter->data.numElements());
	}
	if (inFile.optimizer.step == 0 and inFile.adamM.empty()) return oa::Status::ok();
	if (inFile.adamM.size() != expected or not inFile.adamV.empty()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"Muon checkpoint momentum size does not match the model");
	}
	return oa::Status::ok();
}

oa::Status oa::Muon::loadFrom(oa::Engine& inEngine, const ModelFile& inFile) {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	OA_RETURN_IF_ERROR(validateLoad(inFile));
	// Hyperparams + step always restored (cheap, always present in checkpoint).
	lr_          = inFile.optimizer.lr;
	beta_        = inFile.optimizer.beta1;  // Muon Beta stored in beta1 slot
	eps_         = inFile.optimizer.eps;
	weightDecay_ = inFile.optimizer.weightDecay;
	step_        = static_cast<oa::U64>(inFile.optimizer.step);
	resetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights

	// allocate momentum buffers if first use, then verify sizes line up.
	ensureMomentumBuffers(params_, momentum_);
	if (inFile.adamM.empty()) return oa::Status::ok();

	// drain pending GPU writes to momentum_ before memcpy.
	OA_RETURN_IF_ERROR(context.submitAndWait());

	oa::I64 off = 0;
	for (oa::Usize i = 0; i < momentum_.size(); ++i) {
		oa::I64 n = momentum_[i].numElements();
		const auto bytes = static_cast<oa::U64>(n) * sizeof(oa::F32);
		OA_RETURN_IF_ERROR(oa::EngineResourceAccess::uploadBuffer(
			inEngine, oa::MatrixAccess::descriptor(momentum_[i]), 0,
			inFile.adamM.data() + off, bytes));
		off += n;
	}
	return oa::Status::ok();
}
