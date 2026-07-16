#pragma once

// OaDnn is OA's small semantic graph planner. It does not own matrix storage or
// command recording; it validates mathematical operation graphs and partitions
// them into executable engine contracts implemented by OaBlasLt, dedicated ML
// kernels, or the portable primitive path.

#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/GemmTypes.h>

using OaDnnMatrixId = OaU32;
constexpr OaDnnMatrixId OaInvalidDnnMatrixId = UINT32_MAX;

enum class OaDnnOpType : OaU8 {
	Matmul,
	BiasAdd,
	Relu,
	Gelu,
	Silu,
	Multiply,
	Add,
	RmsNorm,
	FlashAttentionCausal,
	GroupedGemm,
};

enum class OaDnnEngineType : OaU8 {
	Portable,
	BlasLtEpilogue,
	PackedQkv,
	GatedFfn,
	ResidualNorm,
	FlashAttention,
	GroupedMoe,
};

struct OaDnnMatrixDesc {
	OaDnnMatrixId Id = OaInvalidDnnMatrixId;
	OaMatrixShape Shape{};
	OaScalarType Dtype = OaScalarType::Float32;
	bool External = false;
	bool Virtual = false;
};

struct OaDnnOpDesc {
	OaDnnOpType Type = OaDnnOpType::Matmul;
	OaVec<OaDnnMatrixId> Inputs;
	OaVec<OaDnnMatrixId> Outputs;
	// Matmul epilogue semantics are explicit so an engine cannot replay a route
	// with the wrong saved-activation contract.
	OaGemmEpilogue Epilogue = OaGemmEpilogue::None;
	bool Training = true;
};

struct OaDnnPolicy {
	OaU64 MaxWorkspaceBytes = 0;
	bool RequireDeterministic = true;
	bool AllowRecompute = true;
};

struct OaDnnPartition {
	OaDnnEngineType Engine = OaDnnEngineType::Portable;
	OaVec<OaU32> Ops;
	OaVec<OaDnnMatrixId> SavedForBackward;
	OaU64 WorkspaceBytes = 0;
	bool Deterministic = true;
};

struct OaDnnPlan {
	OaVec<OaDnnPartition> Partitions;
	OaU64 GraphHash = 0;
	OaU32 PlannerAbi = 1;
};

class OaDnnGraph {
public:
	[[nodiscard]] OaStatus AddMatrix(const OaDnnMatrixDesc& InMatrix);
	[[nodiscard]] OaStatus AddOp(const OaDnnOpDesc& InOp);
	[[nodiscard]] OaStatus Validate() const;

	[[nodiscard]] const OaDnnMatrixDesc* FindMatrix(OaDnnMatrixId InId) const;
	[[nodiscard]] OaSpan<const OaDnnMatrixDesc> Matrices() const {
		return {Matrices_.Data(), Matrices_.Size()};
	}
	[[nodiscard]] OaSpan<const OaDnnOpDesc> Ops() const {
		return {Ops_.Data(), Ops_.Size()};
	}

private:
	OaVec<OaDnnMatrixDesc> Matrices_;
	OaVec<OaDnnOpDesc> Ops_;
};

class OaDnnPlanner {
public:
	[[nodiscard]] static OaResult<OaDnnPlan> Plan(
		const OaDnnGraph& InGraph, const OaDnnPolicy& InPolicy = {});
};
