// OaMuonRef — Authoritative CPU reference for the Muon optimizer.
//
// Matches Keller Jordan's public reference (Nesterov momentum + Newton-Schulz5)
// and Moonshot's scaled apply (0.2 * sqrt(max(rows, cols))) so AdamW hyperparams
// can be reused. See:
//   https://kellerjordan.github.io/posts/muon/
//   https://arxiv.org/html/2502.16982v1

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Std/Vec.h>
#include <Oa/Ml/Module.h>

namespace OaMuonRef {

/// Official Muon routing: 2D hidden matrices vs AdamW (embed, head, 1D).
struct OaOfficialMuonSplit {
	OaVec<OaParameter*> Muon;
	OaVec<OaParameter*> AdamW;
};

/// Split named parameters per Keller/Moonshot guidance (no fused optimizer needed).
[[nodiscard]] OaOfficialMuonSplit SplitOfficialRouting(OaSpan<const OaNamedParameter> InNamed);

/// True when a named param should use Muon (2D hidden matrix, not embed/head).
[[nodiscard]] bool IsMuonMatrixParam(OaStringView InPath, const OaParameter& InParam);

/// Moonshot per-matrix update scale: 0.2 * sqrt(max(rows, cols)).
[[nodiscard]] OaF32 MoonshotScale(OaU32 InRows, OaU32 InCols, OaF32 InRmsMatch = 0.2f);

/// Newton-Schulz5 orthogonalization. InUpdate/OutOrtho are row-major [rows, cols].
void NewtonSchulz5(
	float* OutOrtho,
	const float* InUpdate,
	OaU32 InRows,
	OaU32 InCols,
	OaI32 InNS5Steps = 5,
	OaF32 InEps = 1e-7f);

/// Full Muon step for a 2D weight matrix (hidden layers).
/// Mutates InOutWeights and InOutMomentum in-place.
void MatrixStep(
	float* InOutWeights,
	float* InOutMomentum,
	const float* InGrads,
	OaU32 InRows,
	OaU32 InCols,
	OaF32 InLr,
	OaF32 InBeta = 0.95f,
	OaF32 InWeightDecay = 0.0f,
	OaF32 InEps = 1e-7f,
	OaI32 InNS5Steps = 5,
	OaF32 InRmsMatch = 0.2f);

/// Momentum + decoupled WD apply for 1D params (biases, norms). No NS5.
/// Official practice uses AdamW here; this is a lightweight fallback.
void VectorStep(
	float* InOutWeights,
	float* InOutMomentum,
	const float* InGrads,
	OaU32 InCount,
	OaF32 InLr,
	OaF32 InBeta = 0.95f,
	OaF32 InWeightDecay = 0.0f);

} // namespace OaMuonRef