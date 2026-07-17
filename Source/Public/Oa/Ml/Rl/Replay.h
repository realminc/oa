#pragma once

#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>

struct OaRlReplayConfig {
	OaU32 Capacity = 0;
	OaMatrixShape ObservationShape;
	OaMatrixShape ActionShape;
	OaScalarType ActionDtype = OaScalarType::Int32;
};

struct OaRlReplayTransition {
	OaMatrix Observation;
	OaMatrix Action;
	OaMatrix NextObservation;
	OaMatrix Reward;
	OaMatrix Terminated;
	OaMatrix Truncated;
};

struct OaRlReplayBatch {
	OaMatrix Observation;
	OaMatrix Action;
	OaMatrix NextObservation;
	OaMatrix Reward;
	OaMatrix Terminated;
	OaMatrix Truncated;
	OaMatrix Index;

	[[nodiscard]] bool IsValid() const noexcept;
};

// Preallocated circular off-policy storage. Appends and deterministic uniform
// sampling stay on the GPU; Size/Cursor are host control metadata and never
// depend on a tensor readback.
class OaRlReplayBuffer {
public:
	[[nodiscard]] static OaResult<OaRlReplayBuffer> Create(
		const OaRlReplayConfig& InConfig);

	[[nodiscard]] OaStatus Append(const OaRlReplayTransition& InTransition);
	[[nodiscard]] OaResult<OaRlReplayBatch> Sample(
		OaU32 InBatchSize, OaU64 InSeed) const;
	void Reset() noexcept;

	[[nodiscard]] bool IsValid() const noexcept;
	[[nodiscard]] bool IsFull() const noexcept { return Size_ == Config_.Capacity; }
	[[nodiscard]] OaU32 Size() const noexcept { return Size_; }
	[[nodiscard]] OaU32 Capacity() const noexcept { return Config_.Capacity; }
	[[nodiscard]] OaU32 Cursor() const noexcept { return Cursor_; }
	[[nodiscard]] const OaRlReplayConfig& Config() const noexcept { return Config_; }

private:
	OaRlReplayConfig Config_;
	OaRlReplayBatch Storage_;
	OaU32 ObservationElements_ = 0;
	OaU32 ActionElements_ = 0;
	OaU32 Size_ = 0;
	OaU32 Cursor_ = 0;
};
