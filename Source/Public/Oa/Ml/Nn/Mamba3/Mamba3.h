#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Ml/Nn.h>
#include <Oa/Core/FnMatrix.h>

// OaMamba3Module — Mamba-3 selective state space model block
//
// Reference: Mamba-3 paper (https://arxiv.org/abs/2603.15569)
// Based on: https://github.com/state-spaces/mamba (mamba_ssm/modules/mamba3.py)
//
class OaMamba3Module : public OaModule {
public:
	OaMamba3Module(
		OaI32 InDModel,
		OaI32 InDState = 128,
		OaI32 InExpand = 2,
		OaI32 InHeadDim = 64,
		OaI32 InNGroups = 1,
		OaF32 InRopeFraction = 0.5f,
		bool InIsMimo = false,
		OaI32 InMimoRank = 4,
		OaF32 InDtMin = 0.001f,
		OaF32 InDtMax = 0.1f,
		OaF32 InDtInitFloor = 1e-4f,
		OaF32 InAFloor = 1e-4f,
		bool InIsOutprojNorm = false
	);

	virtual OaMatrix Forward(const OaMatrix& InInput) override;

	/// Autoregressive single-step for inference (maintains recurrent state across calls).
	virtual OaMatrix Step(const OaMatrix& InInput);

	/// Reset the recurrent decode state to zero for a given batch size.
	virtual void ResetState(OaI32 InBatch);

	// Weight accessors for the fused Empyrealm path (reuses the same weights as the reference Mamba3Module).
	[[nodiscard]] const OaMatrix& InProj() const { return InProj_; }
	[[nodiscard]] const OaMatrix& OutProj() const { return OutProj_; }
	[[nodiscard]] const OaMatrix& NormWeight() const { return NormWeight_; }
	[[nodiscard]] const OaMatrix& DtBias() const { return DtBias_; }
	[[nodiscard]] const OaMatrix& BBias() const { return BBias_; }
	[[nodiscard]] const OaMatrix& CBias() const { return CBias_; }
	[[nodiscard]] const OaMatrix& D() const { return D_; }

	// Config accessors for EmpyrealmCore / general use (no more hardcodes in callers).
	[[nodiscard]] OaI32 GetNHeads() const { return NHeads_; }
	[[nodiscard]] OaI32 GetHeadDim() const { return HeadDim_; }
	[[nodiscard]] OaI32 GetDState() const { return DState_; }
	[[nodiscard]] OaI32 GetExpand() const { return Expand_; }
	[[nodiscard]] OaI32 GetDInner() const { return DInner_; }
	[[nodiscard]] OaI32 GetNumRopeAngles() const { return NumRopeAngles_; }
	[[nodiscard]] OaI32 GetNGroups() const { return NGroups_; }
	[[nodiscard]] bool IsMimo() const { return IsMimo_; }
	[[nodiscard]] OaI32 GetMimoRank() const { return MimoRank_; }
	[[nodiscard]] bool IsOutprojNorm() const { return IsOutprojNorm_; }

	// in_proj split + RMSNorm/discretization shared by Forward and Step.
	struct PreprocOut {
		OaMatrix Ch, Bh;        // [B,L,H,N]
		OaMatrix X, Z;          // [B,L,H,P]
		OaMatrix ADT3, DT3;     // [B,L,H]
		OaMatrix Trap3, Angle3; // [B,L,H], [B,L,A]
		OaMatrix CBias2, BBias2;// [H,N] (or [H,R,N] for mimo; allocated to avoid view-reshapes of params)
	};

	virtual PreprocOut Preprocess(const OaMatrix& InInput, OaI32 InBatch, OaI32 InSeqLen);

protected:
	OaI32 DModel_;
	OaI32 DState_;
	OaI32 Expand_;
	OaI32 HeadDim_;
	OaI32 NGroups_;
	OaI32 DInner_;
	OaI32 NHeads_;
	OaF32 RopeFraction_;
	OaI32 RopeDim_;        // split_tensor_size = floor(d_state * rope_fraction), made even
	OaI32 NumRopeAngles_;  // RopeDim_ / 2  (angle columns in in_proj)
	bool IsMimo_;
	bool IsOutprojNorm_;
	OaI32 MimoRank_;
	OaF32 DtMin_;
	OaF32 DtMax_;
	OaF32 DtInitFloor_;
	OaF32 AFloor_;

	// Parameters
	OaMatrix InProj_;      // [dInProj, d_model]
	OaMatrix DtBias_;      // [n_heads]
	OaMatrix BBias_;       // [n_heads, mimo_rank, d_state]  (SISO: mimo_rank=1)
	OaMatrix CBias_;       // [n_heads, mimo_rank, d_state]
	OaMatrix MimoX_;       // [n_heads, mimo_rank, headdim]
	OaMatrix MimoZ_;       // [n_heads, mimo_rank, headdim]
	OaMatrix MimoO_;       // [n_heads, mimo_rank, headdim]
	OaMatrix D_;           // [n_heads] skip connection
	OaMatrix OutProj_;     // [d_model, d_inner]
	OaMatrix NormWeight_;  // [n_heads, headdim] gated output RMSNorm weight (is_outproj_norm)

	// Recurrent decode state (Step): persists across single-token calls.
	OaMatrix StepSsm_;    // [B, H, P, N]
	OaMatrix StepAngle_;  // [B, H, A]
	OaMatrix StepK_;      // [B, H, N]
	OaMatrix StepV_;      // [B, H, P]

	// Intermediate tensors (kept for step reuse)
	OaMatrix X_;
	OaMatrix Z_;
	OaMatrix Dt_;
	OaMatrix B_;
	OaMatrix C_;
	OaMatrix ALog_;
	OaMatrix States_;
	OaMatrix Y_;
};
