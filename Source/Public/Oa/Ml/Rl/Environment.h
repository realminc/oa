#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Sync.h>

#include <functional>
#include <limits>

class OaEngine;
class OaRlEnvironmentExecution;
class OaRlEnvironmentExecutionAccess;
class OaRlEnvironmentRecordingScope;

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

// Stateful native environment session. The session borrows one OaEngine and
// privately owns its recorder/execution state. Reset/Step remain stateful
// commands; Submit returns the exact completion for every command recorded
// since the previous completion. The public boundary never exposes OaContext
// or queue controls.
class OaRlEnvironment {
public:
	virtual ~OaRlEnvironment();
	OaRlEnvironment(const OaRlEnvironment&) = delete;
	OaRlEnvironment& operator=(const OaRlEnvironment&) = delete;
	OaRlEnvironment(OaRlEnvironment&&) noexcept;
	OaRlEnvironment& operator=(OaRlEnvironment&&) noexcept;

	[[nodiscard]] virtual const OaRlEnvironmentSpec& Spec() const noexcept = 0;
	[[nodiscard]] virtual OaU32 Environments() const noexcept = 0;
	[[nodiscard]] virtual const OaMatrix& Observation() const noexcept = 0;

	// Begin is idempotent while recording. RecordCommands selects the owned
	// execution session only for the callback's dynamic extent, so ambient
	// context state is restored even on failure. A callback failure cancels the
	// whole unsubmitted transaction.
	[[nodiscard]] OaStatus Begin();
	[[nodiscard]] OaStatus RecordCommands(
		const std::function<OaStatus()>& InCommands);
	[[nodiscard]] OaStatus ResetEnvironment(OaU64 InSeed);
	[[nodiscard]] OaResult<OaRlEnvironmentTransition> StepEnvironment(
		const OaMatrix& InAction);
	[[nodiscard]] OaStatus ResetCompleted();

	[[nodiscard]] OaResult<OaEvent> Submit();
	[[nodiscard]] OaStatus Wait(const OaEvent& InEvent);
	// Cancel discards only unsubmitted commands. Close completes an already
	// submitted event, discards unsubmitted commands, and is idempotent.
	[[nodiscard]] OaStatus Cancel();
	[[nodiscard]] OaStatus Close();
	[[nodiscard]] bool IsOpen() const noexcept;
	// True after Begin or the first recorded command and until Submit accepts
	// the transaction or Cancel discards it. Host snapshots must reject this
	// state because recorded writes have not reached the device.
	[[nodiscard]] bool HasActiveRecording() const noexcept;
	[[nodiscard]] bool HasPendingEvent() const noexcept;
	[[nodiscard]] OaU64 SubmissionCount() const noexcept;

protected:
	explicit OaRlEnvironment(OaEngine& InEngine);
	[[nodiscard]] virtual OaStatus RecordResetEnvironment_(OaU64 InSeed) = 0;
	[[nodiscard]] virtual OaResult<OaRlEnvironmentTransition>
		RecordStepEnvironment_(const OaMatrix& InAction) = 0;
	[[nodiscard]] virtual OaStatus RecordResetCompleted_() = 0;
	// Stateful host metadata authored while recording is transactional too.
	// Implementations commit it only once queue submission accepts the batch and
	// roll it back when the whole unsubmitted recording is discarded.
	virtual void CommitRecordedState_() noexcept {}
	virtual void RollbackRecordedState_() noexcept {}

private:
	friend class OaRlEnvironmentExecutionAccess;
	friend class OaRlEnvironmentRecordingScope;
	OaUniquePtr<OaRlEnvironmentExecution> Execution_;
};
