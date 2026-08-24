#pragma once

#include <oa/core/types.h>
#include <oa/ml/nn.h>
#include <oa/core/fnMatrix.h>

namespace oa {

// Mamba3Module — Mamba-3 selective state space model block
//
// Reference: Mamba-3 paper (https://arxiv.org/abs/2603.15569)
// Based on: https://github.com/state-spaces/mamba (mamba_ssm/modules/mamba3.py)
//
class Mamba3Module : public oa::Module {
public:
	Mamba3Module(
		oa::I32 inDModel,
		oa::I32 inDState = 128,
		oa::I32 inExpand = 2,
		oa::I32 inHeadDim = 64,
		oa::I32 inNGroups = 1,
		oa::F32 inRopeFraction = 0.5f,
		bool inIsMimo = false,
		oa::I32 inMimoRank = 4,
		oa::F32 inDtMin = 0.001f,
		oa::F32 inDtMax = 0.1f,
		oa::F32 inDtInitFloor = 1e-4f,
		oa::F32 inAFloor = 1e-4f,
		bool inIsOutprojNorm = false
	);

	virtual oa::Matrix forward(const oa::Matrix& inInput) override;

	/// Autoregressive single-step for inference (maintains recurrent state across calls).
	virtual oa::Matrix step(const oa::Matrix& inInput);

	/// reset the recurrent decode state to zero for a given batch size.
	virtual void resetState(oa::I32 inBatch);

	// weight accessors for the fused empyrealm path (reuses the same weights as the reference Mamba3Module).
	[[nodiscard]] const oa::Matrix& inProj() const { return inProj_; }
	[[nodiscard]] const oa::Matrix& outProj() const { return outProj_; }
	[[nodiscard]] const oa::Matrix& normWeight() const { return normWeight_; }
	[[nodiscard]] const oa::Matrix& dtBias() const { return dtBias_; }
	[[nodiscard]] const oa::Matrix& bBias() const { return bBias_; }
	[[nodiscard]] const oa::Matrix& cBias() const { return cBias_; }
	[[nodiscard]] const oa::Matrix& d() const { return d_; }

	// Config accessors for EmpyrealmCore / general use (no more hardcodes in callers).
	[[nodiscard]] oa::I32 nHeads() const { return nHeads_; }
	[[nodiscard]] oa::I32 headDim() const { return headDim_; }
	[[nodiscard]] oa::I32 dState() const { return dState_; }
	[[nodiscard]] oa::I32 expand() const { return expand_; }
	[[nodiscard]] oa::I32 dInner() const { return dInner_; }
	[[nodiscard]] oa::I32 numRopeAngles() const { return numRopeAngles_; }
	[[nodiscard]] oa::I32 nGroups() const { return nGroups_; }
	[[nodiscard]] bool isMimo() const { return isMimo_; }
	[[nodiscard]] oa::I32 mimoRank() const { return mimoRank_; }
	[[nodiscard]] bool isOutprojNorm() const { return isOutprojNorm_; }

	// in_proj split + RMSNorm/discretization shared by forward and step.
	struct PreprocOut {
		oa::Matrix ch, bh;        // [B,L,G*R,N] (R=1 for SISO)
		oa::Matrix x, z;          // [B,L,H,P]
		oa::Matrix adt3, dt3;     // [B,L,H]
		oa::Matrix trap3, angle3; // [B,L,H], [B,L,A]
		oa::Matrix cBias2, bBias2;// [H,N] (or [H,R,N] for mimo; allocated to avoid view-reshapes of params)
	};

	virtual PreprocOut preprocess(const oa::Matrix& inInput, oa::I32 inBatch, oa::I32 inSeqLen);

protected:
	oa::I32 dModel_;
	oa::I32 dState_;
	oa::I32 expand_;
	oa::I32 headDim_;
	oa::I32 nGroups_;
	oa::I32 dInner_;
	oa::I32 nHeads_;
	oa::F32 ropeFraction_;
	oa::I32 ropeDim_;        // split_tensor_size = floor(d_state * rope_fraction), made even
	oa::I32 numRopeAngles_;  // ropeDim_ / 2  (angle columns in in_proj)
	bool isMimo_;
	bool isOutprojNorm_;
	oa::I32 mimoRank_;
	oa::F32 dtMin_;
	oa::F32 dtMax_;
	oa::F32 dtInitFloor_;
	oa::F32 aFloor_;

	// parameters
	oa::Matrix inProj_;      // [dInProj, d_model]
	oa::Matrix dtBias_;      // [n_heads]
	oa::Matrix bBias_;       // SISO [n_heads,d_state], MIMO [n_heads,rank,d_state]
	oa::Matrix cBias_;       // SISO [n_heads,d_state], MIMO [n_heads,rank,d_state]
	oa::Matrix mimoX_;       // [n_heads, mimo_rank, headdim]
	oa::Matrix mimoZ_;       // [n_heads, mimo_rank, headdim]
	oa::Matrix mimoO_;       // [n_heads, mimo_rank, headdim]
	oa::Matrix d_;           // [n_heads] skip connection
	oa::Matrix outProj_;     // [d_model, d_inner]
	oa::Matrix normWeight_;  // [n_heads, headdim] gated output RMSNorm weight (is_outproj_norm)

	// Recurrent decode state (step): persists across single-token calls.
	oa::Matrix stepSsm_;    // [B, H, P, N]
	oa::Matrix stepAngle_;  // [B, H, A]
	oa::Matrix stepK_;      // SISO [B,H,N], MIMO [B,H,R,N]
	oa::Matrix stepV_;      // SISO [B,H,P], MIMO [B,H,R,P]
};

} // namespace oa
