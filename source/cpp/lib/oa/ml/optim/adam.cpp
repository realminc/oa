#include <oa/ml/optim.h>
#include <oa/ml/modelFile.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/validation.h>
#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/cString.h>

static oa::Matrix getParamGrad(oa::Parameter* inP) {
	return inP->grad();  // live grad (single source of truth: Data's autograd meta)
}

void oa::Adam::step() {
	++step_;

	if (m_.empty()) {
		for (auto* p : params_) {
			m_.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
			v_.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
		}
	}

	auto& ctx = oa::ExecutionSession::getActive();
	for (oa::Usize i = 0; i < params_.size(); ++i) {
		auto* p = params_[i];
		oa::Matrix grad = getParamGrad(p);
		if (!grad.hasStorage()) continue;
		oa::U32 n = static_cast<oa::U32>(p->data.numElements());

		struct { oa::U32 Count; oa::F32 lr; oa::F32 beta1; oa::F32 beta2; oa::F32 eps; oa::U32 step; }
			push{n, lr_, beta1_, beta2_, eps_, static_cast<oa::U32>(step_)};
		oa::BufferAccess access[] = {
			oa::BufferAccess::ReadWrite, oa::BufferAccess::Read,
			oa::BufferAccess::ReadWrite, oa::BufferAccess::ReadWrite
		};
		ctx.add( "Adam", {&p->data, &grad, &m_[i], &v_[i]}, access, &push, sizeof(push), oa::divCeil(n, 256));
	}
}

void oa::Adam::zeroGrad() {
	// Zero the single source of truth GPU-side (self-guarded Fill kernel, never a host
	// memset). See oa::Sgd::zeroGrad.
	for (auto* p : params_) { p->data.zeroGrad(); }
}

// Adam shares the ModelFile optimizer schema with AdamW. weightDecay is unused;
// stored as 0 in the header for round-trip consistency.

oa::Status oa::Adam::saveTo(oa::Engine& inEngine, ModelFile& outFile) const {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	outFile.optimizerPresent = true;
	outFile.optimizer = ModelOptimizerState{};
	constexpr char kType[] = "Adam";
	oa::memcpy(outFile.optimizer.type, kType, sizeof(kType));
	outFile.optimizer.lr    = lr_;
	outFile.optimizer.beta1 = beta1_;
	outFile.optimizer.beta2 = beta2_;
	outFile.optimizer.eps   = eps_;
	outFile.optimizer.step  = static_cast<oa::I64>(step_);

	if (m_.empty()) {
		outFile.optimizer.numParams = 0;
		outFile.adamM.clear();
		outFile.adamV.clear();
		return oa::Status::ok();
	}

	OA_RETURN_IF_ERROR(context.submitAndWait());

	oa::I64 total = 0;
	for (const auto& m : m_) total += m.numElements();
	outFile.optimizer.numParams = static_cast<oa::U64>(total);
	outFile.adamM.resize(total);
	outFile.adamV.resize(total);

	oa::I64 off = 0;
	for (oa::Usize i = 0; i < m_.size(); ++i) {
		oa::I64 n = m_[i].numElements();
		const auto bytes = static_cast<oa::U64>(n) * sizeof(oa::F32);
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			m_[i], outFile.adamM.data() + off, bytes));
		OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
			v_[i], outFile.adamV.data() + off, bytes));
		off += n;
	}
	return oa::Status::ok();
}

oa::Status oa::Adam::validateLoad(const ModelFile& inFile) const {
	if (not inFile.hasOptimizer()
		or oa::strncmp(inFile.optimizer.type, "Adam", sizeof(inFile.optimizer.type)) != 0)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"Adam checkpoint optimizer state is missing or has the wrong type");
	}
	oa::U64 expected = 0;
	for (const auto* parameter : params_) {
		expected += static_cast<oa::U64>(parameter->data.numElements());
	}
	if (inFile.optimizer.step == 0 and inFile.adamM.empty()
		and inFile.adamV.empty()) return oa::Status::ok();
	if (inFile.adamM.size() != expected or inFile.adamV.size() != expected) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"Adam checkpoint moment size does not match the model");
	}
	return oa::Status::ok();
}

oa::Status oa::Adam::loadFrom(oa::Engine& inEngine, const ModelFile& inFile) {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	OA_RETURN_IF_ERROR(validateLoad(inFile));
	lr_    = inFile.optimizer.lr;
	beta1_ = inFile.optimizer.beta1;
	beta2_ = inFile.optimizer.beta2;
	eps_   = inFile.optimizer.eps;
	step_  = static_cast<oa::U64>(inFile.optimizer.step);

	if (m_.empty()) {
		for (auto* p : params_) {
			m_.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
			v_.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
		}
	}
	if (inFile.adamM.empty()) return oa::Status::ok();

	OA_RETURN_IF_ERROR(context.submitAndWait());

	oa::I64 off = 0;
	for (oa::Usize i = 0; i < m_.size(); ++i) {
		oa::I64 n = m_[i].numElements();
		const auto bytes = static_cast<oa::U64>(n) * sizeof(oa::F32);
		OA_RETURN_IF_ERROR(oa::EngineResourceAccess::uploadBuffer(
			inEngine, oa::MatrixAccess::descriptor(m_[i]), 0,
			inFile.adamM.data() + off, bytes));
		OA_RETURN_IF_ERROR(oa::EngineResourceAccess::uploadBuffer(
			inEngine, oa::MatrixAccess::descriptor(v_[i]), 0,
			inFile.adamV.data() + off, bytes));
		off += n;
	}
	return oa::Status::ok();
}
