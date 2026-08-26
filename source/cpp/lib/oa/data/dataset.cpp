// Dataset — deterministic batch iteration.

#include <oa/data/dataset.h>
#include <oa/data/fnDataset.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/random.h>
#include <oa/core/std/utility.h>

namespace oa {

DataLoader::DataLoader(Dataset& inDataset, DataLoaderConfig inConfig)
	: dataset_(inDataset), config_(oa::move(inConfig)) {
	if (config_.batchSize <= 0) {
		OaLogError(oa::LogComponent::Data,
			"DataLoader: batchSize must be positive; using 1");
		config_.batchSize = 1;
	}
	buildIndices();
}

void DataLoader::buildIndices() {
	indices_.clear();
	oa::I64 total = oa::max<oa::I64>(0, dataset_.size());
	indices_.reserve(static_cast<oa::Usize>(total));
	for (oa::I64 i = 0; i < total; ++i) indices_.pushBack(i);
	if (config_.shuffle) {
		oa::U64 seed = config_.seed;
		if (seed == 0) {
			seed = static_cast<oa::U64>(oa::SystemClock::now().nanosecondsSinceEpoch());
		}
		oa::Random rng(seed);
		rng.shuffle(indices_.data(), indices_.size());
	}
}

oa::Optional<DataLoader::Batch> DataLoader::nextBatch() {
	oa::I64 start = currentBatch_ * config_.batchSize;
	if (start >= static_cast<oa::I64>(indices_.size())) return {};

	oa::I64 end = oa::min(start + config_.batchSize, static_cast<oa::I64>(indices_.size()));
	oa::I64 actualBatch = end - start;
	if (config_.dropLast && actualBatch < config_.batchSize) {
		currentBatch_++;
		return {};
	}

	oa::Vec<Dataset::Sample> samples;
	samples.reserve(static_cast<oa::Usize>(actualBatch));
	for (oa::I64 i = 0; i < actualBatch; ++i) {
		auto sample = dataset_.getSample(indices_[static_cast<oa::Usize>(start + i)]);
		samples.pushBack(oa::move(sample));
	}

	currentBatch_++;
	auto collated = oa::FnDataset::collate(
		oa::Span<const Dataset::Sample>(samples.data(), samples.size()));
	if (collated.x.isEmpty()) return {};
	return Batch{oa::move(collated.x), oa::move(collated.y)};
}

void DataLoader::reset() {
	currentBatch_ = 0;
	buildIndices();
}

oa::I64 DataLoader::numBatches() const {
	oa::I64 total = static_cast<oa::I64>(indices_.size());
	oa::I64 batches = total / config_.batchSize;
	if (!config_.dropLast && (total % config_.batchSize) > 0) batches++;
	return batches;
}

} // namespace oa
