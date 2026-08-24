#pragma once

#include <oa/core/status.h>
#include <oa/runtime/gemmTypes.h>
#include <oa/runtime/matmulTypes.h>
#include <oa/runtime/semanticGraphFwd.h>

namespace oa { class ExecutionSession; }
namespace oa {

class Matrix;

struct GemmGraphDesc {
	const Matrix* a = nullptr;
	const Matrix* b = nullptr;
	const Matrix* bias = nullptr;
	Matrix* c = nullptr;
	U32 m = 0;
	U32 n = 0;
	U32 k = 0;
	MatMulPrecision precision = MatMulPrecision::Auto;
	oa::GemmEpilogue epilogue = oa::GemmEpilogue::None;
	oa::MatmulPreference preference = {};
	StringView operation = {};
	U64 opContractHash = 0;
	U32 semanticOp = oa::invalidSemanticOpId;
};

// Internal semantic-GEMM -> executable-graph lowering. This is the only owner
// of matrix layout extraction, plan selection, plan validation, and recording
// for matrix GEMM operations. Public matrix operations describe semantics;
// oa::ExecutionSession merely owns the active recording session.
class GemmGraphLowering {
public:
	// Complete semantic-to-executable MatMulNt recording used by Core. Runtime
	// owns recorder selection, semantic provenance, reshaping and GEMM planning.
	[[nodiscard]] static Result<U32> recordMatMulNt(
		const Matrix& inA,
		const Matrix& inB,
		Matrix& outC,
		U32 inM,
		U32 inN,
		U32 inK,
		MatMulPrecision inPrecision);

	[[nodiscard]] static Status record(
		::oa::ExecutionSession& inContext,
		const GemmGraphDesc& inDesc
	);
};

} // namespace oa
