#pragma once

#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <oa/ml/advantage.h>

namespace oa {

// Fixed-shape categorical on-policy rollout storage. observationShape excludes
// the environment axis; for CartPole it is {4}.
struct RolloutConfig {
	oa::U32 time = 0;
	oa::U32 environments = 0;
	oa::MatrixShape observationShape;
};

// One vector-environment transition. Every matrix remains device-resident.
struct RolloutTransition {
	oa::Matrix observation;       // FP32 [environments, ...observationShape]
	oa::Matrix action;            // Int32 [environments]
	oa::Matrix reward;            // FP32 [environments]
	oa::Matrix value;             // FP32 [environments]
	oa::Matrix nextValue;         // FP32 [environments]
	oa::Matrix logProbability;    // FP32 [environments]
	oa::Matrix terminated;        // UInt8 [environments]
	oa::Matrix truncated;         // UInt8 [environments]
};

// time-major PPO carrier. reshape the leading [time, environments] dimensions
// to create minibatch views without copying storage.
struct RolloutBatch {
	oa::Matrix observation;       // FP32 [time, environments, ...observationShape]
	oa::Matrix action;            // Int32 [time, environments]
	oa::Matrix reward;            // FP32 [time, environments]
	oa::Matrix value;             // FP32 [time, environments]
	oa::Matrix nextValue;         // FP32 [time, environments]
	oa::Matrix oldLogProbability; // FP32 [time, environments]
	oa::Matrix terminated;        // UInt8 [time, environments]
	oa::Matrix truncated;         // UInt8 [time, environments]
	oa::Matrix valid;             // UInt8 [time, environments]
	oa::Matrix advantage;         // FP32 [time, environments]
	oa::Matrix ret;               // FP32 [time, environments]

	[[nodiscard]] bool isValid() const noexcept;
};

class ItRolloutTraining;

class RolloutBuffer {
public:
	RolloutBuffer() = default;
	RolloutBuffer(const RolloutBuffer&) = delete;
	RolloutBuffer& operator=(const RolloutBuffer&) = delete;
	RolloutBuffer(RolloutBuffer&&) noexcept = default;
	RolloutBuffer& operator=(RolloutBuffer&&) noexcept = default;

	[[nodiscard]] static oa::Result<RolloutBuffer> create(
		const RolloutConfig& inConfig);

	// Records one fused GPU append. The CPU cursor is control metadata only; no
	// matrix data is read or copied through the host.
	[[nodiscard]] oa::Status append(const RolloutTransition& inTransition);

	// Requires a complete rollout and records GAE directly into the preallocated
	// advantage/return matrices.
	[[nodiscard]] oa::Status finalize(const GaeConfig& inConfig = {});

	// Begins a new collection cycle and clears valid on the GPU. Previously
	// collected tensors remain allocated and are overwritten in place.
	void reset();

	[[nodiscard]] bool isValid() const noexcept { return batch_.isValid(); }
	[[nodiscard]] bool isFull() const noexcept { return size_ == config_.time; }
	[[nodiscard]] bool isFinalized() const noexcept { return finalized_; }
	[[nodiscard]] oa::U32 size() const noexcept { return size_; }
	[[nodiscard]] oa::U32 capacity() const noexcept { return config_.time; }
	[[nodiscard]] const RolloutConfig& config() const noexcept { return config_; }
	[[nodiscard]] const RolloutBatch& batch() const noexcept { return batch_; }

private:
	friend class ItRolloutTraining;
	// Rolls back only the host cursor/finalization metadata for an unsubmitted
	// collection. The caller owns cancellation of the command transaction;
	// the next reset records the required device-side valid clear.
	void abortUnsubmitted() noexcept;

	RolloutConfig config_;
	RolloutBatch batch_;
	oa::U32 observationElements_ = 0;
	oa::U32 size_ = 0;
	bool finalized_ = false;
};
} // namespace oa
