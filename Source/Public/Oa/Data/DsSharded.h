// OaDsSharded — Sharded byte dataset (OaDataset subclass)
//
// Splits dataset across N files (one per NVMe drive for linear scaling).
// Each shard is independently mmap'd. Works with the generic OaDataLoader
// and supports indexed access (Size, GetItem, GetSample) as well as
// round-robin random sampling via SampleSequence().

#pragma once

#include <Oa/Data/Dataset.h>

class OaDsSharded : public OaDataset {
public:
	/// Open all regular files in InDir as shards
	OaDsSharded(const OaString& InDir, OaI64 InSeqLen);

	// Destructors.
	~OaDsSharded() = default;

	[[nodiscard]] OaI64 Size() const override;
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;

	/// Total bytes across all shards
	[[nodiscard]] OaI64 TotalBytes() const { return TotalBytes_; }

	/// Number of shards
	[[nodiscard]] OaI32 NumShards() const;

	/// Get raw pointer + size for a specific shard
	[[nodiscard]] const OaU8* ShardData(OaI32 InShard) const;
	[[nodiscard]] OaI64 ShardBytes(OaI32 InShard) const;

	/// Sequence length
	[[nodiscard]] OaI64 SeqLen() const { return SeqLen_; }

	/// Sample a random (S+1)-byte sequence. InRng is updated in-place.
	/// Returns pointer into mmap'd shard data (zero-copy).
	[[nodiscard]] const OaU8* SampleSequence(uint64_t& InOutRng) const;

	// Operators.
	OaDsSharded(OaDsSharded&&) noexcept = default;
	OaDsSharded& operator=(OaDsSharded&&) noexcept = default;
	OaDsSharded(const OaDsSharded&) = delete;
	OaDsSharded& operator=(const OaDsSharded&) = delete;

private:
	OaVec<OaMMapByteDataset> Shards_;
	OaI64 SeqLen_ = 0;
	OaI64 TotalBytes_ = 0;
	mutable OaI32 CurrentShard_ = 0;
	OaVec<OaI64> ShardSizes_;  // Precomputed prefix sums for fast index lookup
};
