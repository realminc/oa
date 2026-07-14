// Sharded dataset — multiple files, round-robin across shards
// Each shard is independently mmap'd. With N NVMe drives and N shards,
// read throughput scales linearly. Round-robin ensures even utilization.

#include <Oa/Data/DsSharded.h>
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;

OaDsSharded::OaDsSharded(const OaString& InDir, OaI64 InSeqLen)
	: SeqLen_(InSeqLen) {
	const fs::path dirPath(InDir.StdStr());
	if (!fs::is_directory(dirPath)) {
		fprintf(stderr, "[OaDsSharded] Not a directory: %s\n", InDir.c_str());
		return;
	}

	// Collect all regular files, sorted by name for deterministic ordering
	OaVec<OaString> paths;
	for (const auto& entry : fs::directory_iterator(dirPath)) {
		if (entry.is_regular_file()) {
			paths.PushBack(OaString(entry.path().string()));
		}
	}
	OaStdSort(paths.Begin(), paths.End());

	if (paths.Empty()) {
		fprintf(stderr, "[OaDsSharded] No files found in: %s\n", InDir.c_str());
		return;
	}

	// Open each file as a shard
	for (const auto& path : paths) {
		OaMMapByteDataset shard(path, InSeqLen);
		if (shard.TotalBytes() >= InSeqLen + 1) {
			TotalBytes_ += shard.TotalBytes();
			ShardSizes_.PushBack(shard.Size());
			Shards_.PushBack(std::move(shard));
		}
	}

	fprintf(stderr, "[OaDsSharded] Opened %zu shards, total %.2f GB\n",
		Shards_.Size(), static_cast<double>(TotalBytes_) / (1024.0 * 1024.0 * 1024.0));
}

OaI64 OaDsSharded::Size() const {
	OaI64 total = 0;
	for (OaI64 sz : ShardSizes_) total += sz;
	return total;
}

OaMatrix OaDsSharded::GetItem(OaI64 InIndex) const {
	if (InIndex < 0 || Shards_.Empty()) return OaMatrix();

	// Find which shard contains this index
	OaI64 accumulated = 0;
	for (OaUsize i = 0; i < Shards_.Size(); ++i) {
		OaI64 shardSize = ShardSizes_[i];
		if (InIndex < accumulated + shardSize) {
			return Shards_[i].GetItem(InIndex - accumulated);
		}
		accumulated += shardSize;
	}
	return OaMatrix(); // Index out of range
}

OaI32 OaDsSharded::NumShards() const {
	return static_cast<OaI32>(Shards_.Size());
}

const OaU8* OaDsSharded::ShardData(OaI32 InShard) const {
	if (InShard < 0 || InShard >= static_cast<OaI32>(Shards_.Size())) return nullptr;
	return Shards_[static_cast<OaUsize>(InShard)].RawData();
}

OaI64 OaDsSharded::ShardBytes(OaI32 InShard) const {
	if (InShard < 0 || InShard >= static_cast<OaI32>(Shards_.Size())) return 0;
	return Shards_[static_cast<OaUsize>(InShard)].TotalBytes();
}

const OaU8* OaDsSharded::SampleSequence(uint64_t& InOutRng) const {
	if (Shards_.Empty()) return nullptr;

	// Round-robin shard selection
	OaI32 shard = CurrentShard_;
	CurrentShard_ = (CurrentShard_ + 1) % static_cast<OaI32>(Shards_.Size());

	const OaU8* data = Shards_[static_cast<OaUsize>(shard)].RawData();
	OaI64 shardBytes = Shards_[static_cast<OaUsize>(shard)].TotalBytes();
	OaI64 sampleBytes = SeqLen_ + 1;
	OaI64 maxOffset = shardBytes - sampleBytes;

	if (maxOffset <= 0) return nullptr;

	// LCG step
	InOutRng = InOutRng * 6364136223846793005ULL + 1442695040888963407ULL;
	OaI64 offset = static_cast<OaI64>(InOutRng % static_cast<uint64_t>(maxOffset));

	return data + offset;
}
