#pragma once

// DNN is OA's private semantic graph planner. It does not own matrix storage or
// command recording; it validates mathematical operation graphs and partitions
// them into candidate engine contracts implemented by oa::BlasLt, dedicated ML
// kernels, or the portable primitive path. A plan is analysis metadata until an
// admitted executable lowering explicitly consumes it.

#include <oa/core/matrixShape.h>
#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/runtime/gemmTypes.h>

namespace oa {

class SemanticGraph;

constexpr U32 invalidDnnMatrixId = UINT32_MAX;
constexpr U32 invalidDnnOpId = UINT32_MAX;

enum class DnnOpType : U8 {
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
	GatedMultiply,
	ResidualRmsNorm,
	Portable,
};

enum class DnnEngineType : U8 {
	Portable,
	BlasLtEpilogue,
	QkvProjectionGroup,
	GatedFfn,
	ResidualNorm,
	FlashAttention,
	GroupedMoe,
};

struct DnnMatrixDesc {
	U32 id = invalidDnnMatrixId;
	MatrixShape shape{};
	ScalarType dtype = ScalarType::Float32;
	bool external = false;
	bool isVirtual = false;
};

struct DnnOpDesc {
	// exact source operation identity. Manual graph construction may leave this
	// invalid; addOp then assigns the local SSA ordinal. Automatic semantic
	// capture preserves the canonical oa::U32 here.
	U32 sourceOp = invalidDnnOpId;
	DnnOpType type = DnnOpType::Matmul;
	Vector<U32> inputs;
	Vector<U32> outputs;
	// Matmul epilogue semantics are explicit so an engine cannot replay a route
	// with the wrong saved-activation contract.
	oa::GemmEpilogue epilogue = oa::GemmEpilogue::None;
	bool training = true;
};

struct DnnPolicy {
	U64 maxWorkspaceBytes = 0;
	bool requireDeterministic = true;
	bool allowRecompute = true;
};

struct DnnPartition {
	DnnEngineType engine = DnnEngineType::Portable;
	Vector<U32> ops;
	Vector<U32> savedForBackward;
	U64 workspaceBytes = 0;
	bool deterministic = true;
};

struct DnnPlan {
	Vector<DnnPartition> partitions;
	U64 graphHash = 0;
	U32 plannerAbi = 2;
	U32 sourceOpCount = 0;
	U32 capturedOpCount = 0;
	U32 recognizedPartitionCount = 0;
};

class DnnGraph {
public:
	[[nodiscard]] Status addMatrix(const DnnMatrixDesc& inMatrix);
	[[nodiscard]] Status addOp(const DnnOpDesc& inOp);
	[[nodiscard]] Status validate() const;

	[[nodiscard]] const DnnMatrixDesc* findMatrix(U32 inId) const;
	[[nodiscard]] Span<const DnnMatrixDesc> matrices() const {
		return {matrices_.data(), matrices_.size()};
	}
	[[nodiscard]] Span<const DnnOpDesc> ops() const {
		return {ops_.data(), ops_.size()};
	}

private:
	Vector<DnnMatrixDesc> matrices_;
	Vector<DnnOpDesc> ops_;
};

class DnnPlanner {
public:
	[[nodiscard]] static Result<DnnPlan> plan(
		const DnnGraph& inGraph, const DnnPolicy& inPolicy = {});
	// capture the matrix regions of OA's canonical semantic graph. Unsupported
	// operations remain portable partitions; this overload never asks model or
	// autograd code to author a second graph.
	[[nodiscard]] static Result<DnnPlan> plan(
		const SemanticGraph& inGraph, const DnnPolicy& inPolicy = {});
};

} // namespace oa
