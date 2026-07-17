#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>

#include <limits>

// Structural domains shared by native OA environments and interoperability
// adapters. Shapes exclude the leading environment-batch dimension; rank zero
// therefore represents one scalar per environment.
enum class OaRlSpaceKind : OaU8 {
	Box,
	Discrete,
	Binary,
};

struct OaRlFieldSpec {
	OaString Name;
	OaRlSpaceKind Kind = OaRlSpaceKind::Box;
	OaMatrixShape Shape;
	OaScalarType Dtype = OaScalarType::Float32;
	OaF64 Minimum = -std::numeric_limits<OaF64>::infinity();
	OaF64 Maximum = std::numeric_limits<OaF64>::infinity();
	OaI64 Cardinality = 0;

	[[nodiscard]] static OaRlFieldSpec Box(
		OaStringView InName,
		OaMatrixShape InShape,
		OaScalarType InDtype = OaScalarType::Float32,
		OaF64 InMinimum = -std::numeric_limits<OaF64>::infinity(),
		OaF64 InMaximum = std::numeric_limits<OaF64>::infinity()
	);
	[[nodiscard]] static OaRlFieldSpec Discrete(
		OaStringView InName,
		OaI64 InCardinality,
		OaScalarType InDtype = OaScalarType::Int32
	);
	[[nodiscard]] static OaRlFieldSpec Binary(
		OaStringView InName,
		OaMatrixShape InShape = {},
		OaScalarType InDtype = OaScalarType::UInt8
	);

	[[nodiscard]] OaStatus ValidateDefinition() const;
	[[nodiscard]] OaI64 ElementsPerEnvironment() const noexcept;
	[[nodiscard]] OaResult<OaMatrixShape> BatchedShape(
		OaU32 InEnvironments
	) const;
	[[nodiscard]] OaStatus ValidateMatrix(
		const OaMatrix& InMatrix,
		OaU32 InEnvironments
	) const;
};

// One single-agent environment schema. It is intentionally a value contract,
// not a virtual environment hierarchy: native GPU environments and Python
// adapters can expose the same metadata without sharing execution machinery.
struct OaRlEnvironmentSpec {
	OaRlFieldSpec Observation;
	OaRlFieldSpec Action;
	OaRlFieldSpec Reward;
	OaRlFieldSpec Terminated;
	OaRlFieldSpec Truncated;

	[[nodiscard]] OaStatus ValidateDefinition() const;
	[[nodiscard]] OaStatus ValidateReset(
		const OaMatrix& InObservation,
		OaU32 InEnvironments
	) const;

	[[nodiscard]] OaStatus ValidateAction(
		const OaMatrix& InAction,
		OaU32 InEnvironments
	) const;

	[[nodiscard]] OaStatus ValidateTransition(
		const OaMatrix& InObservation,
		const OaMatrix& InAction,
		const OaMatrix& InNextObservation,
		const OaMatrix& InReward,
		const OaMatrix& InTerminated,
		const OaMatrix& InTruncated,
		OaU32 InEnvironments
	) const;
};

struct OaRlEnvironmentTransition {
	OaMatrix Observation;
	OaMatrix NextObservation;
	OaMatrix Reward;
	OaMatrix Terminated;
	OaMatrix Truncated;

	[[nodiscard]] bool IsValid() const noexcept {
		return !Observation.IsEmpty() && !NextObservation.IsEmpty()
			&& !Reward.IsEmpty() && !Terminated.IsEmpty()
			&& !Truncated.IsEmpty();
	}
};

// Minimal native execution boundary shared by collectors and adapters. It does
// not prescribe simulation internals, rendering, info dictionaries or device
// placement; those remain environment-owned.
class OaRlEnvironment {
public:
	virtual ~OaRlEnvironment() = default;
	[[nodiscard]] virtual const OaRlEnvironmentSpec& Spec() const noexcept = 0;
	[[nodiscard]] virtual OaU32 Environments() const noexcept = 0;
	[[nodiscard]] virtual const OaMatrix& Observation() const noexcept = 0;
	[[nodiscard]] virtual OaStatus ResetEnvironment(OaU64 InSeed) = 0;
	[[nodiscard]] virtual OaResult<OaRlEnvironmentTransition> StepEnvironment(
		const OaMatrix& InAction) = 0;
	[[nodiscard]] virtual OaStatus ResetCompleted() = 0;
};
