// oa::FnDataset — unified dataset operations implementation.

#include <oa/data/fnDataset.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <oa/core/std/algo.h>
#include <oa/core/std/chrono.h>
#include <oa/core/std/random.h>

namespace oa {

namespace FnDataset {

namespace {

bool isValidSampleMatrix(const oa::Matrix& inMatrix) {
	return !inMatrix.isEmpty() && inMatrix.hasStorage() &&
		inMatrix.rank() < OA_MAX_TENSOR_DIMS && inMatrix.byteSize() > 0;
}

oa::MatrixShape batchedShape(const oa::MatrixShape& inItemShape, oa::I64 inBatchSize) {
	oa::MatrixShape shape = inItemShape;
	for (oa::I32 i = shape.rank; i > 0; --i) {
		shape.dims[static_cast<oa::Usize>(i)] =
			shape.dims[static_cast<oa::Usize>(i) - 1];
	}
	shape.dims[0] = inBatchSize;
	++shape.rank;
	return shape;
}

bool validateLike(const oa::Matrix& inMatrix, const oa::Matrix& inReference) {
	return isValidSampleMatrix(inMatrix) &&
		inMatrix.getShape() == inReference.getShape() &&
		inMatrix.getDtype() == inReference.getDtype() &&
		inMatrix.byteSize() == inReference.byteSize();
}

} // namespace

// ============================================================================
// index shuffling and splitting
// ============================================================================

void shuffle(oa::Vector<oa::I64>& inOutIndices, oa::U64 inSeed) {
	if (inSeed == 0) {
		inSeed = static_cast<oa::U64>(
			oa::SystemClock::now().nanosecondsSinceEpoch());
	}
	oa::Random rng(inSeed);
	rng.shuffle(inOutIndices.data(), inOutIndices.size());
}

SplitResult randomSplit(oa::I64 inTotalSize, oa::F32 inTrainRatio, oa::F32 inValRatio, oa::U64 inSeed) {
	SplitResult result;
	if (inTotalSize <= 0) return result;
	inTrainRatio = oa::clamp(inTrainRatio, 0.0f, 1.0f);
	inValRatio = oa::clamp(inValRatio, 0.0f, 1.0f - inTrainRatio);

	oa::Vector<oa::I64> indices(static_cast<oa::Usize>(inTotalSize));
	for (oa::I64 index = 0; index < inTotalSize; ++index) {
		indices[static_cast<oa::Usize>(index)] = index;
	}
	shuffle(indices, inSeed);

	oa::I64 trainCount = static_cast<oa::I64>(static_cast<oa::F32>(inTotalSize) * inTrainRatio);
	oa::I64 valCount   = static_cast<oa::I64>(static_cast<oa::F32>(inTotalSize) * inValRatio);
	oa::I64 testCount  = inTotalSize - trainCount - valCount;
	if (testCount < 0) {
		trainCount += testCount; // absorb deficit into train
		testCount = 0;
	}
	if (valCount < 0) {
		trainCount += valCount;
		valCount = 0;
	}

	result.train.reserve(static_cast<oa::Usize>(trainCount));
	result.val.reserve(static_cast<oa::Usize>(valCount));
	result.test.reserve(static_cast<oa::Usize>(testCount));

	for (oa::I64 i = 0; i < trainCount; ++i)
		result.train.pushBack(indices[static_cast<oa::Usize>(i)]);
	for (oa::I64 i = 0; i < valCount; ++i)
		result.val.pushBack(indices[static_cast<oa::Usize>(trainCount + i)]);
	for (oa::I64 i = 0; i < testCount; ++i)
		result.test.pushBack(indices[static_cast<oa::Usize>(trainCount + valCount + i)]);

	return result;
}

// ============================================================================
// Batch collation
// ============================================================================

Batch collate(oa::Span<const Dataset::Sample> inSamples) {
	if (inSamples.size() == 0) return {};

	const auto& first = inSamples[0];
	if (!isValidSampleMatrix(first.x)) {
		OaLogError(oa::LogComponent::Data,
			"oa::FnDataset::collate: first X is empty, unstored, or has maximum rank");
		return {};
	}

	const bool hasLabel = first.hasLabel();
	if (hasLabel && !isValidSampleMatrix(first.y)) {
		OaLogError(oa::LogComponent::Data,
			"oa::FnDataset::collate: first Y is invalid");
		return {};
	}

	for (oa::Usize i = 1; i < inSamples.size(); ++i) {
		const auto& sample = inSamples[i];
		if (!validateLike(sample.x, first.x)) {
			OaLogError(oa::LogComponent::Data,
				"oa::FnDataset::collate: X shape, dtype, or storage mismatch at sample {}", i);
			return {};
		}
		if (sample.hasLabel() != hasLabel ||
			(hasLabel && !validateLike(sample.y, first.y))) {
			OaLogError(oa::LogComponent::Data,
				"oa::FnDataset::collate: Y presence, shape, dtype, or storage mismatch at sample {}", i);
			return {};
		}
	}

	oa::I64 actualBatch = static_cast<oa::I64>(inSamples.size());

	// X batch: prepend batch dimension
	oa::MatrixShape xShape = batchedShape(first.x.getShape(), actualBatch);
	auto xBatch = oa::FnMatrix::empty(xShape, first.x.getDtype());
	if (!xBatch.hasStorage()) return {};

	// Y batch: optional
	oa::Matrix yBatch;
	if (hasLabel) {
		oa::MatrixShape yShape = batchedShape(first.y.getShape(), actualBatch);
		yBatch = oa::FnMatrix::empty(yShape, first.y.getDtype());
		if (!yBatch.hasStorage()) return {};
	}

	oa::Usize xItemBytes = static_cast<oa::Usize>(first.x.byteSize());
	oa::Usize yItemBytes = hasLabel ? static_cast<oa::Usize>(first.y.byteSize()) : 0;
	for (oa::Usize i = 0; i < inSamples.size(); ++i) {
		const auto& sample = inSamples[i];
		oa::memcpy(xBatch.dataAs<oa::U8>() + i * xItemBytes,
			sample.x.dataAs<const oa::U8>(), xItemBytes);
		if (hasLabel) {
			oa::memcpy(yBatch.dataAs<oa::U8>() + i * yItemBytes,
				sample.y.dataAs<const oa::U8>(), yItemBytes);
		}
	}

	return Batch{oa::move(xBatch), oa::move(yBatch)};
}

} // namespace FnDataset

} // namespace oa
