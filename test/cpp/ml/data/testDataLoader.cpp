#include <oa/data/dataset.h>
#include <oa/data/dsSubset.h>
#include <gtest/gtest.h>
#include "../../oaTest.h"

namespace {

oa::Matrix u8Matrix(std::initializer_list<oa::U8> inValues, oa::MatrixShape inShape) {
	oa::Vector<oa::U8> values;
	values.reserve(inValues.size());
	for (oa::U8 value : inValues) values.pushBack(value);
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(values.data(), values.size()),
		inShape, oa::ScalarType::UInt8);
}

class TestDataset final : public oa::Dataset {
public:
	explicit TestDataset(bool inMismatch = false) : mismatch_(inMismatch) {}

	[[nodiscard]] oa::I64 size() const override { return 3; }

	[[nodiscard]] oa::Matrix getItem(oa::I64 inIndex) const override {
		return getSample(inIndex).x;
	}

	[[nodiscard]] Sample getSample(oa::I64 inIndex) const override {
		if (inIndex < 0 || inIndex >= size()) return {};
		if (mismatch_ && inIndex == 1) {
			return Sample(
				u8Matrix({3, 4, 5}, oa::MatrixShape{3}),
				u8Matrix({1}, oa::MatrixShape{1}));
		}
		const oa::U8 base = static_cast<oa::U8>(inIndex * 2);
		return Sample(
			u8Matrix({base, static_cast<oa::U8>(base + 1)}, oa::MatrixShape{2}),
			u8Matrix({static_cast<oa::U8>(inIndex)}, oa::MatrixShape{1}));
	}

private:
	bool mismatch_;
};

} // namespace

TEST(DataLoader, CollatesConsistentSamples) {
	TestDataset dataset;
	oa::DataLoader loader(dataset, {
		.batchSize = 2,
		.shuffle = false,
	});

	auto batch = loader.nextBatch();
	ASSERT_TRUE(batch.hasValue());
	expectShape(batch->x, {2, 2});
	expectShape(batch->y, {2, 1});
	std::array<oa::U8, 4> xBytes{};
	std::array<oa::U8, 2> yBytes{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		batch->x, xBytes.data(), xBytes.size()).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		batch->y, yBytes.data(), yBytes.size()).isOk());
	EXPECT_EQ(xBytes, (std::array<oa::U8, 4>{0, 1, 2, 3}));
	EXPECT_EQ(yBytes, (std::array<oa::U8, 2>{0, 1}));

	auto tail = loader.nextBatch();
	ASSERT_TRUE(tail.hasValue());
	expectShape(tail->x, {1, 2});
	EXPECT_FALSE(loader.nextBatch().hasValue());
}

TEST(DataLoader, RejectsMismatchedSampleShapes) {
	TestDataset dataset(true);
	oa::DataLoader loader(dataset, {
		.batchSize = 2,
		.shuffle = false,
	});
	EXPECT_FALSE(loader.nextBatch().hasValue());
}

TEST(DataLoader, DropLastAdvancesToEpochEnd) {
	TestDataset dataset;
	oa::DataLoader loader(dataset, {
		.batchSize = 2,
		.shuffle = false,
		.dropLast = true,
	});
	ASSERT_TRUE(loader.nextBatch().hasValue());
	EXPECT_FALSE(loader.nextBatch().hasValue());
	EXPECT_FALSE(loader.nextBatch().hasValue());
}

TEST(DataLoader, SubsetMapsSamplesAndRejectsBounds) {
	TestDataset dataset;
	const std::array<oa::I64, 2> indices{2, 0};
	oa::DsSubset subset(
		dataset,
		oa::Span<const oa::I64>(indices.data(), indices.size()));

	ASSERT_EQ(subset.size(), 2);
	ASSERT_EQ(subset.indices().size(), 2);
	EXPECT_EQ(subset.indices()[0], 2);
	EXPECT_EQ(subset.indices()[1], 0);
	EXPECT_EQ(&subset.parent(), &dataset);

	auto first = subset.getSample(0);
	ASSERT_TRUE(first.hasLabel());
	std::array<oa::U8, 2> xBytes{};
	std::array<oa::U8, 1> yBytes{};
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		first.x, xBytes.data(), xBytes.size()).isOk());
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		first.y, yBytes.data(), yBytes.size()).isOk());
	EXPECT_EQ(xBytes, (std::array<oa::U8, 2>{4, 5}));
	EXPECT_EQ(yBytes, (std::array<oa::U8, 1>{2}));

	EXPECT_TRUE(subset.getItem(-1).isEmpty());
	EXPECT_TRUE(subset.getItem(2).isEmpty());
	EXPECT_TRUE(subset.getSample(2).x.isEmpty());
}
