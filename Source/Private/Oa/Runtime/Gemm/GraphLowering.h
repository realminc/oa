#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

class OaContext;
class OaMatrix;

struct OaGemmGraphDesc {
	const OaMatrix* A = nullptr;
	const OaMatrix* B = nullptr;
	const OaMatrix* Bias = nullptr;
	OaMatrix* C = nullptr;
	OaU32 M = 0;
	OaU32 N = 0;
	OaU32 K = 0;
	OaMatMulPrecision Precision = OaMatMulPrecision::Auto;
	OaGemmEpilogue Epilogue = OaGemmEpilogue::None;
	OaStringView Operation = {};
	OaU64 OperationContractHash = 0;
	OaSemanticOperationId SemanticOperation = OaInvalidSemanticOperationId;
};

// Internal semantic-GEMM -> executable-graph lowering. This is the only owner
// of matrix layout extraction, plan selection, plan validation, and recording
// for matrix GEMM operations. Public matrix operations describe semantics;
// OaContext merely owns the active recording session.
class OaGemmGraphLowering {
public:
	// Complete semantic-to-executable MatMulNt recording used by Core. Runtime
	// owns recorder selection, semantic provenance, reshaping and GEMM planning.
	[[nodiscard]] static OaResult<OaSemanticOperationId> RecordMatMulNt(
		const OaMatrix& InA,
		const OaMatrix& InB,
		OaMatrix& OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaMatMulPrecision InPrecision);

	[[nodiscard]] static OaStatus Record(
		OaContext& InContext,
		const OaGemmGraphDesc& InDesc
	);
};
