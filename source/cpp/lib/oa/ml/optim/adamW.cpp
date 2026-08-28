#include <oa/ml/optim.h>
#include <oa/ml/fnOptim.h>
#include <oa/ml/modelFile.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/fnmatrix/fnMatrixInternal.h>
#include <oa/core/log.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/cString.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/executionSession.h>

static oa::Matrix getParamGrad(oa::Parameter* inP) {
	return inP->grad();  // live grad (single source of truth: Data's autograd meta)
}

static void ensureMomentBuffers(oa::Vector<oa::Parameter*>& inParams,
	oa::Vector<oa::Matrix>& inOutM, oa::Vector<oa::Matrix>& inOutV)
{
	if (not inOutM.empty()) return;
	for (auto* p : inParams) {
		inOutM.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
		inOutV.pushBack(oa::FnMatrix::zeros(p->data.getShape()));
	}
}

void oa::AdamW::setLr(oa::F32 inLr) {
	lr_ = inLr;
	if (not graphState_.hasStorage() or graphState_.data() == nullptr) return;
	OA_ASSERT(graphStateEngine_ != nullptr);

	// replay reads LR from the mutable state buffer rather than baked push
	// constants. oa::ItTraining waits for the replay before callbacks run, so this
	// host update cannot race the preceding optimizer dispatch.
	(void)oa::EngineResourceAccess::uploadBuffer(*graphStateEngine_,
		oa::MatrixAccess::descriptor(graphState_), sizeof(oa::U32), &lr_, sizeof(oa::F32));
}

void oa::AdamW::step() {
	++step_;

	// Lazy-initialize moment buffers on first step
	ensureMomentBuffers(params_, m_, v_);
	// Lazy-allocate + seed fp32 master weights for any low-precision (bf16) params.
	prepareMasters();

	auto& ctx = oa::ExecutionSession::getActive();
	if (not graphState_.hasStorage()) {
		// replay metadata is intentionally host-visible: callbacks update LR and
		// the completed step between submissions. Model matrices remain governed
		// by the engine's device-local placement policy.
		graphState_ = oa::FnMatrix::empty(
			oa::MatrixShape{6}, oa::ScalarType::UInt32, oa::MemoryPlacement::HostUpload);
		graphStateEngine_ = &ctx.engine();
	}
	OA_ASSERT(graphStateEngine_ == &ctx.engine());
	const oa::Bool replayState = ctx.isStableResourceFrameActive()
		and graphState_.hasStorage();
	if (replayState) {
		// The recorded graph advances this word on the GPU before any parameter
		// update. Seeding the previous completed step preserves eager semantics on
		// the first execution and lets subsequent command-buffer replays advance
		// without a host rewrite.
		oa::U32 state[6] = {static_cast<oa::U32>(step_ - 1U)};
		const oa::F32 scalars[] = {lr_, beta1_, beta2_, eps_, weightDecay_};
		oa::memcpy(state + 1, scalars, sizeof(scalars));
		(void)oa::EngineResourceAccess::uploadBuffer(
			ctx.engine(), oa::MatrixAccess::descriptor(graphState_), 0,
			state, sizeof(state));
		oa::FnOptim::adamWAdvanceGraphState(graphState_);
	}

	// Any low-precision param routes through the fp32-master per-param path. The
	// fused 4-param fast path assumes fp32 m/v under a single DTYPE, which breaks
	// once a weight is bf16 (it would read the fp32 moments as bf16) — so skip it.
	bool anyMaster = false;
	for (oa::Usize i = 0; i < params_.size(); ++i) {
		if (hasMaster(i)) { anyMaster = true; break; }
	}

	// grad() returns a handle sharing the live grad buffer. The param set needs
	// stable addresses, so hold the four handles in locals; private dispatch
	// recording (inside AdamWStepMany) snapshots their vkBuf_ at record time and
	// keeps the buffer owners alive, so these locals can safely die when step()
	// returns.
	if (not anyMaster and params_.size() == 4) {
		oa::Matrix grads4[4] = {
			params_[0]->grad(), params_[1]->grad(),
			params_[2]->grad(), params_[3]->grad(),
		};
		if (grads4[0].hasStorage() and grads4[1].hasStorage()
			and grads4[2].hasStorage() and grads4[3].hasStorage()) {
			oa::FnOptim::AdamWParamSet sets[4] = {
				{&params_[0]->data, &m_[0], &v_[0], &grads4[0]},
				{&params_[1]->data, &m_[1], &v_[1], &grads4[1]},
				{&params_[2]->data, &m_[2], &v_[2], &grads4[2]},
				{&params_[3]->data, &m_[3], &v_[3], &grads4[3]},
			};
			if (replayState) {
				oa::FnOptim::adamWStepManyGraph(
					oa::Span<const oa::FnOptim::AdamWParamSet>(sets, 4), graphState_);
			} else {
				oa::FnOptim::adamWStepMany(
					oa::Span<const oa::FnOptim::AdamWParamSet>(sets, 4),
					lr_, beta1_, beta2_, eps_, weightDecay_, static_cast<oa::I32>(step_));
			}
			return;
		}
	}

	// apply AdamW update to each parameter using oa::FnOptim stateless function.
	// For bf16 params the base helpers redirect the math onto the fp32 master
	// (grad upcast) and re-derive the bf16 weight afterwards; fp32 params update
	// in place (MasterOrData → Data, MasterGrad → grad, WritebackMaster → no-op).
	for (oa::Usize i = 0; i < params_.size(); ++i) {
		auto* p = params_[i];
		oa::Matrix grad = getParamGrad(p);
		if (!grad.hasStorage()) continue;

		oa::Matrix gradUse = masterGrad(i, grad);
		if (replayState) {
			oa::FnOptim::adamWStepGraph(
				masterOrData(i), m_[i], v_[i], gradUse, graphState_);
		} else {
			oa::FnOptim::adamWStep(
				masterOrData(i), m_[i], v_[i], gradUse,
				lr_, beta1_, beta2_, eps_, weightDecay_, static_cast<oa::I32>(step_));
		}
		writebackMaster(i);   // fp32 master → bf16 weight (no-op for fp32 params)
	}
}

void oa::AdamW::zeroGrad() {
	// Zero the single source of truth (each param's autograd grad) GPU-side. grad()
	// returns a handle sharing the live buffer, so MultiFill on these copies fills the
	// real GPU buffers. Never a host memset (silent no-op on GPU → grads accumulate
	// across steps → divergence).
	//
	// Fast path: batch up to 4 grads into a single MultiMatrixFill dispatch.
	oa::Vector<oa::Matrix> grads;
	grads.reserve(4);
	for (auto* p : params_) {
		oa::Matrix g = p->grad();
		if (g.hasStorage()) grads.pushBack(g);
	}
	for (oa::Usize i = 0; i < grads.size(); i += 4) {
		oa::Usize end = oa::min(i + 4, grads.size());
		oa::FnMatrix::multiFill(oa::Span<oa::Matrix>(grads.data() + i, end - i), 0.0F);
	}
}

// ─── Persistence ──────────────────────────────────────────────────────────

oa::Status oa::AdamW::saveTo(oa::Engine& inEngine, ModelFile& outFile) const {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	// header (hyperparams + step count). numParams is total flat element count.
	outFile.optimizerPresent = true;
	outFile.optimizer = ModelOptimizerState{};
	constexpr char kType[] = "AdamW";
	oa::memcpy(outFile.optimizer.type, kType, sizeof(kType));
	outFile.optimizer.lr = lr_;
	outFile.optimizer.beta1 = beta1_;
	outFile.optimizer.beta2 = beta2_;
	outFile.optimizer.eps = eps_;
	outFile.optimizer.weightDecay = weightDecay_;
	outFile.optimizer.step = static_cast<oa::I64>(step_);

	// m_/v_ may not be allocated yet (no step() called). then nothing to save.
	if (m_.empty()) {
		outFile.optimizer.numParams = 0;
		outFile.adamM.clear();
		outFile.adamV.clear();
		return oa::Status::ok();
	}

	// drain pending GPU writes to m_/v_ so the memcpy sees the latest state.
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

oa::Status oa::AdamW::validateLoad(const ModelFile& inFile) const {
	if (not inFile.hasOptimizer()
		or oa::strncmp(inFile.optimizer.type, "AdamW", sizeof(inFile.optimizer.type)) != 0)
	{
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"AdamW checkpoint optimizer state is missing or has the wrong type");
	}
	oa::U64 expected = 0;
	for (const auto* parameter : params_) {
		expected += static_cast<oa::U64>(parameter->data.numElements());
	}
	if (inFile.optimizer.step == 0 and inFile.adamM.empty()
		and inFile.adamV.empty()) return oa::Status::ok();
	if (inFile.adamM.size() != expected or inFile.adamV.size() != expected) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"AdamW checkpoint moment size does not match the model");
	}
	return oa::Status::ok();
}

oa::Status oa::AdamW::loadFrom(oa::Engine& inEngine, const ModelFile& inFile) {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	OA_RETURN_IF_ERROR(validateLoad(inFile));
	// Hyperparams + step always restored (cheap, always present in checkpoint).
	lr_          = inFile.optimizer.lr;
	beta1_       = inFile.optimizer.beta1;
	beta2_       = inFile.optimizer.beta2;
	eps_         = inFile.optimizer.eps;
	weightDecay_ = inFile.optimizer.weightDecay;
	step_        = static_cast<oa::U64>(inFile.optimizer.step);
	resetMasterSeed();  // re-seed fp32 masters from the reloaded (bf16) weights

	// allocate moment buffers if first use, then verify sizes line up.
	ensureMomentBuffers(params_, m_, v_);
	if (inFile.adamM.empty()) return oa::Status::ok();

	// drain pending GPU writes to m_/v_ before memcpy.
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
