#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>

// ─── empyrealm fused A·dt term ───────────────────────────────────────────────
// forward: ADT = min(-heavy_tail(ddA), -aFloor) * dt   (fused EmpyrealmAdt kernel).
// Both inputs are [B*S, H]; backward is pure elementwise (EmpyrealmAdtBwd kernel).
namespace oa {

class GradEmpyrealmAdt final : public oa::GradNode {
public:
	explicit GradEmpyrealmAdt(oa::F32 inAFloor) noexcept : aFloor_(inAFloor) {}
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& ddA = saved(0);
		const oa::Matrix& dt  = saved(1);
		auto g = oa::FnMatrix::empyrealmAdtBwd(inDOut, ddA, dt, aFloor_);
		if (outDIn.size() > 0) outDIn[0] = g.dDdA;
		if (outDIn.size() > 1) outDIn[1] = g.dDt;
	}
private:
	oa::F32 aFloor_;
};

// ─── empyrealm fused dt term ─────────────────────────────────────────────────
// forward: DT = clamp(softplus(x), dt_min, dt_max)   (fused EmpyrealmDt kernel).
// backward is pure elementwise (EmpyrealmDtBwd kernel).
class GradEmpyrealmDt final : public oa::GradNode {
public:
	GradEmpyrealmDt(oa::F32 inDtMin, oa::F32 inDtMax) noexcept : dtMin_(inDtMin), dtMax_(inDtMax) {}
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& x = saved(0);
		if (outDIn.size() > 0) outDIn[0] = oa::FnMatrix::empyrealmDtBwd(inDOut, x, dtMin_, dtMax_);
	}
private:
	oa::F32 dtMin_;
	oa::F32 dtMax_;
};

// ─── Mamba-3 fused preprocess (split + rmsnorm + dt + adt in one kernel) ─────
// forward: mamba3Preprocess(projected, dtBias, config) → {X, Z, bh, ch, DT, ADT, trap, angle}
// backward: Mamba3PreprocessBwd → {dProjected, dDtBias}
//
// Multi-output lazy-merge pattern: 8 thin grad nodes (one per output) share a
// SharedState. Each saves its output gradient and decrements a counter. The
// last one (counter→0) dispatches the fused backward and returns the result.
// The other 7 return empty matrices (skipped by the tape).
class GradMamba3Preprocess final : public oa::GradNode {
public:
	struct SharedState {
		oa::I32 counter = 8;
		oa::Matrix projected, dtBias;
		oa::Matrix dZ, dX, dBh, dCh, dDt, dAdt, dTrap, dAngle;
		oa::Matrix dProjected, dDtBias;
		oa::Mamba3PreprocessConfig config;
	};

	oa::SharedPtr<SharedState> state_;
	oa::I32 outputIndex_ = 0;  // 0=z, 1=x, 2=bh, 3=ch, 4=dt, 5=adt, 6=trap, 7=angle

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		switch (outputIndex_) {
			case 0: state_->dZ = inDOut; break;
			case 1: state_->dX = inDOut; break;
			case 2: state_->dBh = inDOut; break;
			case 3: state_->dCh = inDOut; break;
			case 4: state_->dDt = inDOut; break;
			case 5: state_->dAdt = inDOut; break;
			case 6: state_->dTrap = inDOut; break;
			case 7: state_->dAngle = inDOut; break;
		}
		oa::I32 remaining = --state_->counter;
		if (remaining == 0) {
			auto grads = oa::FnMatrix::mamba3PreprocessBwd(
				state_->projected, state_->dtBias,
				state_->dZ, state_->dX, state_->dBh, state_->dCh,
				state_->dDt, state_->dAdt, state_->dTrap, state_->dAngle,
				state_->config);
			state_->dProjected = grads.dProjected;
			state_->dDtBias = grads.dDtBias;
			if (outDIn.size() > 0) outDIn[0] = grads.dProjected;
			if (outDIn.size() > 1) outDIn[1] = grads.dDtBias;
		}
	}
};

// ─── empyrealm fused preprocess (renamed copy of Mamba3Preprocess) ───────────

class GradEmpyrealmPreprocess final : public oa::GradNode {
public:
	oa::SharedPtr<oa::GradMamba3Preprocess::SharedState> state_;
	oa::I32 outputIndex_ = 0;

	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		switch (outputIndex_) {
			case 0: state_->dZ = inDOut; break;
			case 1: state_->dX = inDOut; break;
			case 2: state_->dBh = inDOut; break;
			case 3: state_->dCh = inDOut; break;
			case 4: state_->dDt = inDOut; break;
			case 5: state_->dAdt = inDOut; break;
			case 6: state_->dTrap = inDOut; break;
			case 7: state_->dAngle = inDOut; break;
		}
		oa::I32 remaining = --state_->counter;
		if (remaining == 0) {
			auto grads = oa::FnMatrix::empyrealmPreprocessBwd(
				state_->projected, state_->dtBias,
				state_->dZ, state_->dX, state_->dBh, state_->dCh,
				state_->dDt, state_->dAdt, state_->dTrap, state_->dAngle,
				state_->config);
			state_->dProjected = grads.dProjected;
			state_->dDtBias = grads.dDtBias;
			if (outDIn.size() > 0) outDIn[0] = grads.dProjected;
			if (outDIn.size() > 1) outDIn[1] = grads.dDtBias;
		}
	}
};

// ─── Mamba-3 SISO selective scan ─────────────────────────────────────────────

class GradMamba3Siso final : public oa::GradNode {
public:
	oa::SsmConfig config_{};
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& c      = saved(0);
		const oa::Matrix& b      = saved(1);
		const oa::Matrix& x      = saved(2);
		const oa::Matrix& z      = saved(3);
		const oa::Matrix& adt    = saved(4);
		const oa::Matrix& dt     = saved(5);
		const oa::Matrix& trap   = saved(6);
		const oa::Matrix& angle  = saved(7);
		const oa::Matrix& cbias  = saved(8);
		const oa::Matrix& bbias  = saved(9);
		const oa::Matrix& d      = saved(10);
		// normalize dout shape: downstream views/reshapes (e.g. before out_proj MatMul,
		// or in the LM wrapper) may deliver a flat/3D/2D tensor sharing the siso output's
		// autograd meta. The Bwd host + kernel expect the original 4D output shape.
		oa::MatrixShape expected = oa::MatrixShape{config_.batch, config_.seqLen, config_.nHeads, config_.headDim};
		oa::Matrix dOut4 = inDOut;
		if (inDOut.getShape() != expected && inDOut.numElements() == (oa::I64)config_.batch * config_.seqLen * config_.nHeads * config_.headDim) {
			dOut4 = inDOut.reshape(expected);
		}
		auto g = oa::FnMatrix::mamba3SisoBwd(dOut4, c, b, x, z, adt, dt, trap, angle,
			cbias, bbias, d, config_);
		if (outDIn.size() > 0)  outDIn[0]  = g.dC;
		if (outDIn.size() > 1)  outDIn[1]  = g.dB;
		if (outDIn.size() > 2)  outDIn[2]  = g.dX;
		if (outDIn.size() > 3)  outDIn[3]  = g.dZ;
		if (outDIn.size() > 4)  outDIn[4]  = g.dAdt;
		if (outDIn.size() > 5)  outDIn[5]  = g.dDt;
		if (outDIn.size() > 6)  outDIn[6]  = g.dTrap;
		if (outDIn.size() > 7)  outDIn[7]  = g.dAngle;
		if (outDIn.size() > 8)  outDIn[8]  = g.dCBias;
		if (outDIn.size() > 9)  outDIn[9]  = g.dBBias;
		if (outDIn.size() > 10) outDIn[10] = g.dD;
	}
};

// ─── Mamba-3 MIMO selective scan ────────────────────────────────────────────
// MIMO is one shared selective state, not R independent SISO states. Keep its
// adjoint explicit so lowering cannot erase cross-rank coupling or move the
// rankwise normalization after contraction.
class GradMamba3Mimo final : public oa::GradNode {
public:
	oa::SsmConfig config_{};
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		oa::MatrixShape expected{config_.batch, config_.seqLen,
			config_.nHeads, config_.headDim};
		oa::Matrix dOut4 = inDOut;
		if (inDOut.getShape() != expected &&
			inDOut.numElements() == static_cast<oa::I64>(config_.batch) *
				config_.seqLen * config_.nHeads * config_.headDim)
		{
			dOut4 = inDOut.reshape(expected);
		}
		auto g = oa::FnMatrix::mamba3MimoBwd(
			dOut4, saved(0), saved(1), saved(2), saved(3), saved(4),
			saved(5), saved(6), saved(7), saved(8), saved(9), saved(10),
			saved(11), saved(12), saved(13), saved(14), config_);
		if (outDIn.size() > 0) outDIn[0] = g.dC;
		if (outDIn.size() > 1) outDIn[1] = g.dB;
		if (outDIn.size() > 2) outDIn[2] = g.dX;
		if (outDIn.size() > 3) outDIn[3] = g.dZ;
		if (outDIn.size() > 4) outDIn[4] = g.dAdt;
		if (outDIn.size() > 5) outDIn[5] = g.dDt;
		if (outDIn.size() > 6) outDIn[6] = g.dTrap;
		if (outDIn.size() > 7) outDIn[7] = g.dAngle;
		if (outDIn.size() > 8) outDIn[8] = g.dCBias;
		if (outDIn.size() > 9) outDIn[9] = g.dBBias;
		if (outDIn.size() > 10) outDIn[10] = g.dD;
		if (outDIn.size() > 11) outDIn[11] = g.dMimoX;
		if (outDIn.size() > 12) outDIn[12] = g.dMimoZ;
		if (outDIn.size() > 13) outDIn[13] = g.dMimoO;
		if (outDIn.size() > 14) outDIn[14] = g.dNormWeight;
	}
};

// The current empyrealm scan is namespace-distinct but mathematically matches
// the ungated Mamba-3 scan. Keep its adjoint namespace-distinct as well so the
// semantic graph records EmpyrealmSisoBwd rather than Mamba3SisoBwd.
class GradEmpyrealmSiso final : public oa::GradNode {
public:
	oa::SsmConfig config_{};
	void backward(const oa::Matrix& inDOut, oa::Vec<oa::Matrix>& outDIn) override {
		const oa::Matrix& c      = saved(0);
		const oa::Matrix& b      = saved(1);
		const oa::Matrix& x      = saved(2);
		const oa::Matrix& z      = saved(3);
		const oa::Matrix& adt    = saved(4);
		const oa::Matrix& dt     = saved(5);
		const oa::Matrix& trap   = saved(6);
		const oa::Matrix& angle  = saved(7);
		const oa::Matrix& cbias  = saved(8);
		const oa::Matrix& bbias  = saved(9);
		const oa::Matrix& d      = saved(10);
		oa::MatrixShape expected = oa::MatrixShape{
			config_.batch, config_.seqLen, config_.nHeads, config_.headDim};
		oa::Matrix dOut4 = inDOut;
		if (inDOut.getShape() != expected &&
			inDOut.numElements() == static_cast<oa::I64>(config_.batch) *
				config_.seqLen * config_.nHeads * config_.headDim)
		{
			dOut4 = inDOut.reshape(expected);
		}
		auto g = oa::FnMatrix::empyrealmSisoBwd(
			dOut4, c, b, x, z, adt, dt, trap, angle, cbias, bbias, d, config_);
		if (outDIn.size() > 0)  outDIn[0]  = g.dC;
		if (outDIn.size() > 1)  outDIn[1]  = g.dB;
		if (outDIn.size() > 2)  outDIn[2]  = g.dX;
		if (outDIn.size() > 3)  outDIn[3]  = g.dZ;
		if (outDIn.size() > 4)  outDIn[4]  = g.dAdt;
		if (outDIn.size() > 5)  outDIn[5]  = g.dDt;
		if (outDIn.size() > 6)  outDIn[6]  = g.dTrap;
		if (outDIn.size() > 7)  outDIn[7]  = g.dAngle;
		if (outDIn.size() > 8)  outDIn[8]  = g.dCBias;
		if (outDIn.size() > 9)  outDIn[9]  = g.dBBias;
		if (outDIn.size() > 10) outDIn[10] = g.dD;
	}
};

} // namespace oa
