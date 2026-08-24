#pragma once

#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/runtime/event.h>

#include <functional>
#include <limits>

namespace oa {

class Engine;
class EnvironmentExecution;
class EnvironmentExecutionAccess;
class EnvironmentRecordingScope;

// Structural domains shared by native OA environments and interoperability
// adapters. Shapes exclude the leading environment-batch dimension; rank zero
// therefore represents one scalar per environment.
enum class EnvironmentSpaceKind : oa::U8 {
	Box,
	Discrete,
	Binary,
};

struct EnvironmentSpace {
	oa::String name;
	EnvironmentSpaceKind kind = EnvironmentSpaceKind::Box;
	oa::MatrixShape shape;
	oa::ScalarType dtype = oa::ScalarType::Float32;
	oa::F64 minimum = -std::numeric_limits<oa::F64>::infinity();
	oa::F64 maximum = std::numeric_limits<oa::F64>::infinity();
	oa::I64 cardinality = 0;

	[[nodiscard]] static EnvironmentSpace box(
		oa::StringView inName,
		oa::MatrixShape inShape,
		oa::ScalarType inDtype = oa::ScalarType::Float32,
		oa::F64 inMinimum = -std::numeric_limits<oa::F64>::infinity(),
		oa::F64 inMaximum = std::numeric_limits<oa::F64>::infinity()
	);
	[[nodiscard]] static EnvironmentSpace discrete(
		oa::StringView inName,
		oa::I64 inCardinality,
		oa::ScalarType inDtype = oa::ScalarType::Int32
	);
	[[nodiscard]] static EnvironmentSpace binary(
		oa::StringView inName,
		oa::MatrixShape inShape = {},
		oa::ScalarType inDtype = oa::ScalarType::UInt8
	);

	[[nodiscard]] oa::Status validateDefinition() const;
	[[nodiscard]] oa::I64 elementsPerEnvironment() const noexcept;
	[[nodiscard]] oa::Result<oa::MatrixShape> batchedShape(
		oa::U32 inEnvironments
	) const;
	[[nodiscard]] oa::Status validateMatrix(
		const oa::Matrix& inMatrix,
		oa::U32 inEnvironments
	) const;
};

// One single-agent environment schema. It is intentionally a value contract,
// not a virtual environment hierarchy: native GPU environments and Python
// adapters can expose the same metadata without sharing execution machinery.
struct EnvironmentSpec {
	EnvironmentSpace observation;
	EnvironmentSpace action;
	EnvironmentSpace reward;
	EnvironmentSpace terminated;
	EnvironmentSpace truncated;

	[[nodiscard]] oa::Status validateDefinition() const;
	[[nodiscard]] oa::Status validateReset(
		const oa::Matrix& inObservation,
		oa::U32 inEnvironments
	) const;

	[[nodiscard]] oa::Status validateAction(
		const oa::Matrix& inAction,
		oa::U32 inEnvironments
	) const;

	[[nodiscard]] oa::Status validateTransition(
		const oa::Matrix& inObservation,
		const oa::Matrix& inAction,
		const oa::Matrix& inNextObservation,
		const oa::Matrix& inReward,
		const oa::Matrix& inTerminated,
		const oa::Matrix& inTruncated,
		oa::U32 inEnvironments
	) const;
};

struct EnvironmentTransition {
	oa::Matrix observation;
	oa::Matrix nextObservation;
	oa::Matrix reward;
	oa::Matrix terminated;
	oa::Matrix truncated;

	[[nodiscard]] bool isValid() const noexcept {
		return !observation.isEmpty() && !nextObservation.isEmpty()
			&& !reward.isEmpty() && !terminated.isEmpty()
			&& !truncated.isEmpty();
	}
};

// Stateful native environment session. The session borrows one Engine and
// privately owns its recorder/execution state. reset/step remain stateful
// commands; submit returns the exact completion for every command recorded
// since the previous completion. The public boundary never exposes context
// or queue controls.
class Environment {
public:
	virtual ~Environment();
	Environment(const Environment&) = delete;
	Environment& operator=(const Environment&) = delete;
	Environment(Environment&&) noexcept;
	Environment& operator=(Environment&&) noexcept;

	[[nodiscard]] virtual const EnvironmentSpec& spec() const noexcept = 0;
	[[nodiscard]] virtual oa::U32 environments() const noexcept = 0;
	[[nodiscard]] virtual const oa::Matrix& observation() const noexcept = 0;

	// begin is idempotent while recording. recordCommands selects the owned
	// execution session only for the callback's dynamic extent, so ambient
	// context state is restored even on failure. A callback failure cancels the
	// whole unsubmitted transaction.
	[[nodiscard]] oa::Status begin();
	[[nodiscard]] oa::Status recordCommands(
		const std::function<oa::Status()>& inCommands);
	[[nodiscard]] oa::Status reset(oa::U64 inSeed);
	[[nodiscard]] oa::Result<EnvironmentTransition> step(
		const oa::Matrix& inAction);
	[[nodiscard]] oa::Status resetCompleted();

	[[nodiscard]] oa::Result<oa::Event> submit();
	[[nodiscard]] oa::Status wait(const oa::Event& inEvent);
	// cancel discards only unsubmitted commands. close completes an already
	// submitted event, discards unsubmitted commands, and is idempotent.
	[[nodiscard]] oa::Status cancel();
	[[nodiscard]] oa::Status close();
	[[nodiscard]] bool isOpen() const noexcept;
	// True after begin or the first recorded command and until submit accepts
	// the transaction or cancel discards it. Host snapshots must reject this
	// state because recorded writes have not reached the device.
	[[nodiscard]] bool hasActiveRecording() const noexcept;
	[[nodiscard]] bool hasPendingEvent() const noexcept;
	[[nodiscard]] oa::U64 submissionCount() const noexcept;

protected:
	explicit Environment(oa::Engine& inEngine);
	[[nodiscard]] virtual oa::Status recordReset_(oa::U64 inSeed) = 0;
	[[nodiscard]] virtual oa::Result<EnvironmentTransition>
		recordStep_(const oa::Matrix& inAction) = 0;
	[[nodiscard]] virtual oa::Status recordResetCompleted_() = 0;
	// Stateful host metadata authored while recording is transactional too.
	// Implementations commit it only once queue submission accepts the batch and
	// roll it back when the whole unsubmitted recording is discarded.
	virtual void commitRecordedState_() noexcept {}
	virtual void rollbackRecordedState_() noexcept {}

private:
	friend class EnvironmentExecutionAccess;
	friend class EnvironmentRecordingScope;
	oa::UniquePtr<EnvironmentExecution> execution_;
};

} // namespace oa
