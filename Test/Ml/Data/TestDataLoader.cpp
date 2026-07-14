#include <Oa/Data/Dataset.h>
#include <gtest/gtest.h>
#include "../../OaTest.h"

namespace {

OaMatrix U8Matrix(std::initializer_list<OaU8> InValues, OaMatrixShape InShape) {
	OaVec<OaU8> values;
	values.Reserve(InValues.size());
	for (OaU8 value : InValues) values.PushBack(value);
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(values.Data(), values.Size()),
		InShape, OaScalarType::UInt8);
}

class OaTestDataset final : public OaDataset {
public:
	explicit OaTestDataset(bool InMismatch = false) : Mismatch_(InMismatch) {}

	[[nodiscard]] OaI64 Size() const override { return 3; }

	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override {
		return GetSample(InIndex).X;
	}

	[[nodiscard]] Sample GetSample(OaI64 InIndex) const override {
		if (InIndex < 0 || InIndex >= Size()) return {};
		if (Mismatch_ && InIndex == 1) {
			return Sample(
				U8Matrix({3, 4, 5}, OaMatrixShape{3}),
				U8Matrix({1}, OaMatrixShape{1}));
		}
		const OaU8 base = static_cast<OaU8>(InIndex * 2);
		return Sample(
			U8Matrix({base, static_cast<OaU8>(base + 1)}, OaMatrixShape{2}),
			U8Matrix({static_cast<OaU8>(InIndex)}, OaMatrixShape{1}));
	}

private:
	bool Mismatch_;
};

} // namespace

TEST(DataLoader, CollatesConsistentSamples) {
	OaTestDataset dataset;
	OaDataLoader loader(dataset, {
		.BatchSize = 2,
		.Shuffle = false,
	});

	auto batch = loader.NextBatch();
	ASSERT_TRUE(batch.has_value());
	OaExpectShape(batch->X, {2, 2});
	OaExpectShape(batch->Y, {2, 1});
	std::array<OaU8, 4> xBytes{};
	std::array<OaU8, 2> yBytes{};
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		batch->X, xBytes.data(), xBytes.size()).IsOk());
	ASSERT_TRUE(OaFnMatrix::CopyToHost(
		batch->Y, yBytes.data(), yBytes.size()).IsOk());
	EXPECT_EQ(xBytes, (std::array<OaU8, 4>{0, 1, 2, 3}));
	EXPECT_EQ(yBytes, (std::array<OaU8, 2>{0, 1}));

	auto tail = loader.NextBatch();
	ASSERT_TRUE(tail.has_value());
	OaExpectShape(tail->X, {1, 2});
	EXPECT_FALSE(loader.NextBatch().has_value());
}

TEST(DataLoader, RejectsMismatchedSampleShapes) {
	OaTestDataset dataset(true);
	OaDataLoader loader(dataset, {
		.BatchSize = 2,
		.Shuffle = false,
	});
	EXPECT_FALSE(loader.NextBatch().has_value());
}

TEST(DataLoader, DropLastAdvancesToEpochEnd) {
	OaTestDataset dataset;
	OaDataLoader loader(dataset, {
		.BatchSize = 2,
		.Shuffle = false,
		.DropLast = true,
	});
	ASSERT_TRUE(loader.NextBatch().has_value());
	EXPECT_FALSE(loader.NextBatch().has_value());
	EXPECT_FALSE(loader.NextBatch().has_value());
}
