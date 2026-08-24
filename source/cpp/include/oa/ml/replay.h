#pragma once

#include <oa/core/matrix.h>
#include <oa/core/status.h>

namespace oa {

struct ReplayConfig {
	oa::U32 capacity = 0;
	oa::MatrixShape observationShape;
	oa::MatrixShape actionShape;
	oa::ScalarType actionDtype = oa::ScalarType::Int32;
};

struct ReplayTransition {
	oa::Matrix observation;
	oa::Matrix action;
	oa::Matrix nextObservation;
	oa::Matrix reward;
	oa::Matrix terminated;
	oa::Matrix truncated;
};

struct ReplayBatch {
	oa::Matrix observation;
	oa::Matrix action;
	oa::Matrix nextObservation;
	oa::Matrix reward;
	oa::Matrix terminated;
	oa::Matrix truncated;
	oa::Matrix index;

	[[nodiscard]] bool isValid() const noexcept;
};

// Preallocated circular off-policy storage. Appends and deterministic uniform
// sampling stay on the GPU; size/cursor are host control metadata and never
// depend on a tensor readback.
class ReplayBuffer {
public:
	[[nodiscard]] static oa::Result<ReplayBuffer> create(
		const ReplayConfig& inConfig);

	[[nodiscard]] oa::Status append(const ReplayTransition& inTransition);
	[[nodiscard]] oa::Result<ReplayBatch> sample(
		oa::U32 inBatchSize, oa::U64 inSeed) const;
	void reset() noexcept;

	[[nodiscard]] bool isValid() const noexcept;
	[[nodiscard]] bool isFull() const noexcept { return size_ == config_.capacity; }
	[[nodiscard]] oa::U32 size() const noexcept { return size_; }
	[[nodiscard]] oa::U32 capacity() const noexcept { return config_.capacity; }
	[[nodiscard]] oa::U32 cursor() const noexcept { return cursor_; }
	[[nodiscard]] const ReplayConfig& config() const noexcept { return config_; }

private:
	ReplayConfig config_;
	ReplayBatch storage_;
	oa::U32 observationElements_ = 0;
	oa::U32 actionElements_ = 0;
	oa::U32 size_ = 0;
	oa::U32 cursor_ = 0;
};

} // namespace oa
