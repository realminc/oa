#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <Oa/Ml/Rl/Gae.h>

// Fixed-shape categorical on-policy rollout storage. ObservationShape excludes
// the environment axis; for CartPole it is {4}.
struct OaRlRolloutConfig {
	OaU32 Time = 0;
	OaU32 Environments = 0;
	OaMatrixShape ObservationShape;
};

// One vector-environment transition. Every matrix remains device-resident.
struct OaRlTransition {
	OaMatrix Observation;       // FP32 [Environments, ...ObservationShape]
	OaMatrix Action;            // Int32 [Environments]
	OaMatrix Reward;            // FP32 [Environments]
	OaMatrix Value;             // FP32 [Environments]
	OaMatrix NextValue;         // FP32 [Environments]
	OaMatrix LogProbability;    // FP32 [Environments]
	OaMatrix Terminated;        // UInt8 [Environments]
	OaMatrix Truncated;         // UInt8 [Environments]
};

// Time-major PPO carrier. Reshape the leading [Time, Environments] dimensions
// to create minibatch views without copying storage.
struct OaRlRolloutBatch {
	OaMatrix Observation;       // FP32 [Time, Environments, ...ObservationShape]
	OaMatrix Action;            // Int32 [Time, Environments]
	OaMatrix Reward;            // FP32 [Time, Environments]
	OaMatrix Value;             // FP32 [Time, Environments]
	OaMatrix NextValue;         // FP32 [Time, Environments]
	OaMatrix OldLogProbability; // FP32 [Time, Environments]
	OaMatrix Terminated;        // UInt8 [Time, Environments]
	OaMatrix Truncated;         // UInt8 [Time, Environments]
	OaMatrix Valid;             // UInt8 [Time, Environments]
	OaMatrix Advantage;         // FP32 [Time, Environments]
	OaMatrix Return;            // FP32 [Time, Environments]

	[[nodiscard]] bool IsValid() const noexcept;
};

class OaRlRolloutBuffer {
public:
	OaRlRolloutBuffer() = default;
	OaRlRolloutBuffer(const OaRlRolloutBuffer&) = delete;
	OaRlRolloutBuffer& operator=(const OaRlRolloutBuffer&) = delete;
	OaRlRolloutBuffer(OaRlRolloutBuffer&&) noexcept = default;
	OaRlRolloutBuffer& operator=(OaRlRolloutBuffer&&) noexcept = default;

	[[nodiscard]] static OaResult<OaRlRolloutBuffer> Create(
		const OaRlRolloutConfig& InConfig);

	// Records one fused GPU append. The CPU cursor is control metadata only; no
	// matrix data is read or copied through the host.
	[[nodiscard]] OaStatus Append(const OaRlTransition& InTransition);

	// Requires a complete rollout and records GAE directly into the preallocated
	// Advantage/Return matrices.
	[[nodiscard]] OaStatus Finalize(const OaGaeConfig& InConfig = {});

	// Begins a new collection cycle and clears Valid on the GPU. Previously
	// collected tensors remain allocated and are overwritten in place.
	void Reset();

	[[nodiscard]] bool IsValid() const noexcept { return Batch_.IsValid(); }
	[[nodiscard]] bool IsFull() const noexcept { return Size_ == Config_.Time; }
	[[nodiscard]] bool IsFinalized() const noexcept { return Finalized_; }
	[[nodiscard]] OaU32 Size() const noexcept { return Size_; }
	[[nodiscard]] OaU32 Capacity() const noexcept { return Config_.Time; }
	[[nodiscard]] const OaRlRolloutConfig& Config() const noexcept { return Config_; }
	[[nodiscard]] const OaRlRolloutBatch& Batch() const noexcept { return Batch_; }

private:
	OaRlRolloutConfig Config_;
	OaRlRolloutBatch Batch_;
	OaU32 ObservationElements_ = 0;
	OaU32 Size_ = 0;
	bool Finalized_ = false;
};
