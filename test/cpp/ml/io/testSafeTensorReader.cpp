#include "../../oaTest.h"

#include <oa/ml/transferWeights.h>

#include <atomic>
#include <chrono>

namespace {

std::atomic<oa::U64> GSafeTensorTestSequence{0};

oa::Path makeTestDirectory() {
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	const std::string sequence =
		std::to_string(static_cast<unsigned long long>(++GSafeTensorTestSequence));
	const std::string timestamp = std::to_string(static_cast<long long>(tick));
	const oa::String name = oa::String("oa_safetensor_")
		+ oa::String(sequence.data(), sequence.size()) + "_"
		+ oa::String(timestamp.data(), timestamp.size());
	return oa::Paths::temp() / oa::Path(name);
}

oa::Status writeSafeTensor(
	const oa::Path& inPath,
	oa::StringView inHeader,
	oa::Span<const oa::U8> inPayload
) {
	const oa::U64 headerSize = static_cast<oa::U64>(inHeader.size());
	oa::Vec<oa::U8> bytes(static_cast<oa::Usize>(8 + headerSize + inPayload.size()));
	oa::memcpy(bytes.data(), &headerSize, sizeof(headerSize));
	oa::memcpy(bytes.data() + 8, inHeader.data(), static_cast<oa::Usize>(headerSize));
	if (!inPayload.empty()) {
		oa::memcpy(bytes.data() + 8 + headerSize, inPayload.data(), inPayload.size());
	}
	return oa::Filesystem::writeBinary(inPath, {bytes.data(), bytes.size()});
}

class TransferWeightsTest : public ::testing::Test {
protected:
	void SetUp() override {
		directory_ = makeTestDirectory();
		ASSERT_TRUE(oa::Filesystem::createDirectories(directory_).isOk());
	}

	void TearDown() override {
		(void)oa::Filesystem::removeDirectory(directory_, true);
	}

	oa::Path file(const char* inName) const {
		return directory_ / oa::Path(inName);
	}

	oa::Path directory_;
};

} // namespace

TEST_F(TransferWeightsTest, OpensAndExposesCheckedEntrySpans) {
	const oa::String header =
		"{\"__metadata__\":{\"format\":\"pt\"},"
		"\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
	const oa::F32 values[] = {1.0F, 2.0F, 3.0F, 4.0F};
	const auto payload = oa::Span<const oa::U8>(
		reinterpret_cast<const oa::U8*>(values), sizeof(values));
	ASSERT_TRUE(writeSafeTensor(file("valid.safetensors"), header, payload).isOk());

	auto sourceResult = oa::openWeightSource(file("valid.safetensors"));
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString().cStr();
	auto& source = *sourceResult.getValue();
	EXPECT_EQ(source.list().size(), 1u);

	const auto* entry = source.find("weight");
	ASSERT_NE(entry, nullptr);
	EXPECT_EQ(entry->dtype, oa::ScalarType::Float32);
	EXPECT_EQ(entry->elementCount, 4u);
	EXPECT_EQ(entry->byteSize, sizeof(values));

	auto bytes = source.bytes("weight");
	ASSERT_TRUE(bytes.isOk()) << bytes.getStatus().toString().cStr();
	ASSERT_EQ(bytes->size(), sizeof(values));
	EXPECT_EQ(oa::memcmp(bytes->data(), values, sizeof(values)), 0);
	EXPECT_EQ(source.metadata().at("format"), "pt");
}

TEST_F(TransferWeightsTest, ConvertsFloat32ToBfloat16) {
	const oa::String header =
		"{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
	const oa::F32 values[] = {1.0F, -2.0F};
	ASSERT_TRUE(writeSafeTensor(
		file("convert.safetensors"), header,
		{reinterpret_cast<const oa::U8*>(values), sizeof(values)}).isOk());

	auto sourceResult = oa::openWeightSource(file("convert.safetensors"));
	ASSERT_TRUE(sourceResult.isOk());
	auto& source = *sourceResult.getValue();
	oa::U16 converted[2] = {};
	ASSERT_TRUE(source.read(
		"weight",
		{reinterpret_cast<oa::U8*>(converted), sizeof(converted)},
		oa::ScalarType::BFloat16).isOk());
	EXPECT_EQ(converted[0], 0x3F80u);
	EXPECT_EQ(converted[1], 0xC000u);
}

TEST_F(TransferWeightsTest, ExecutesIdentityTransposeConcatAndSliceMappings) {
	const oa::String header =
		"{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
		"\"b\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[16,24]}}";
	const oa::F32 values[] = {1, 2, 3, 4, 5, 6};
	ASSERT_TRUE(writeSafeTensor(file("mapped.safetensors"), header,
		{reinterpret_cast<const oa::U8*>(values), sizeof(values)}).isOk());
	auto sourceResult = oa::openWeightSource(file("mapped.safetensors"));
	ASSERT_TRUE(sourceResult.isOk());

	oa::WeightMap map;
	map.architecture = "mapping-test";
	map.mappings = {
		{.sources = {"a"}, .target = "identity", .targetShape = {2, 2},
			.targetDtype = oa::ScalarType::Float32, .transform = oa::WeightTransform::Identity},
		{.sources = {"a"}, .target = "transpose", .targetShape = {2, 2},
			.targetDtype = oa::ScalarType::Float32, .transform = oa::WeightTransform::Transpose2D},
		{.sources = {"a", "b"}, .target = "concat", .targetShape = {2, 3},
			.targetDtype = oa::ScalarType::Float32, .transform = oa::WeightTransform::Concat,
			.concatAxis = 1},
		{.sources = {"a"}, .target = "slice", .targetShape = {2, 1},
			.targetDtype = oa::ScalarType::Float32, .transform = oa::WeightTransform::Slice,
			.slice = {.axis = 1, .begin = 1, .length = 1}},
	};
	oa::ModelFile model;
	auto reportResult = oa::transferWeights(*sourceResult.getValue(), map, model);
	ASSERT_TRUE(reportResult.isOk()) << reportResult.getStatus().toString().cStr();
	EXPECT_EQ(reportResult->sourceWeights, 2u);
	EXPECT_EQ(reportResult->usedSourceWeights, 2u);
	EXPECT_EQ(reportResult->outputWeights, 4u);
	EXPECT_TRUE(reportResult->unusedSources.empty());

	const oa::F32 identity[] = {1, 2, 3, 4};
	const oa::F32 transpose[] = {1, 3, 2, 4};
	const oa::F32 concat[] = {1, 2, 5, 3, 4, 6};
	const oa::F32 slice[] = {2, 4};
	ASSERT_NE(model.weightPtr("identity"), nullptr);
	EXPECT_EQ(oa::memcmp(model.weightPtr("identity"), identity, sizeof(identity)), 0);
	EXPECT_EQ(oa::memcmp(model.weightPtr("transpose"), transpose, sizeof(transpose)), 0);
	EXPECT_EQ(oa::memcmp(model.weightPtr("concat"), concat, sizeof(concat)), 0);
	EXPECT_EQ(oa::memcmp(model.weightPtr("slice"), slice, sizeof(slice)), 0);
}

TEST_F(TransferWeightsTest, EnforcesCompleteSourceCoverage) {
	const oa::String header =
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"unused\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const oa::U8 values[] = {1, 2};
	ASSERT_TRUE(writeSafeTensor(file("coverage.safetensors"), header, values).isOk());
	auto sourceResult = oa::openWeightSource(file("coverage.safetensors"));
	ASSERT_TRUE(sourceResult.isOk());
	oa::WeightMap map;
	map.architecture = "coverage-test";
	map.mappings.pushBack({.sources = {"a"}, .target = "a", .targetShape = {1},
		.targetDtype = oa::ScalarType::UInt8});
	oa::ModelFile model;
	auto strictResult = oa::transferWeights(*sourceResult.getValue(), map, model);
	ASSERT_TRUE(strictResult.isError());
	EXPECT_EQ(strictResult.getStatus().getCode(), oa::StatusCode::FailedPrecondition);

	map.requireAllSourceWeights = false;
	auto permissiveResult = oa::transferWeights(*sourceResult.getValue(), map, model);
	ASSERT_TRUE(permissiveResult.isOk());
	ASSERT_EQ(permissiveResult->unusedSources.size(), 1u);
	EXPECT_EQ(permissiveResult->unusedSources[0], "unused");
}

TEST_F(TransferWeightsTest, OpensIndexedShardPackageAsOneWeightSource) {
	const oa::U8 a[] = {11};
	const oa::U8 b[] = {22};
	ASSERT_TRUE(writeSafeTensor(file("model-00001-of-00002.safetensors"),
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", a).isOk());
	ASSERT_TRUE(writeSafeTensor(file("model-00002-of-00002.safetensors"),
		"{\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", b).isOk());
	const oa::String index =
		"{\"metadata\":{\"total_size\":\"2\"},\"weight_map\":{"
		"\"a\":\"model-00001-of-00002.safetensors\","
		"\"b\":\"model-00002-of-00002.safetensors\"}}";
	ASSERT_TRUE(oa::Filesystem::writeText(file("model.safetensors.index.json"), index).isOk());

	auto sourceResult = oa::openWeightSource(directory_);
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString().cStr();
	auto& source = *sourceResult.getValue();
	ASSERT_EQ(source.list().size(), 2u);
	ASSERT_NE(source.find("a"), nullptr);
	ASSERT_NE(source.find("b"), nullptr);
	auto aBytes = source.bytes("a");
	auto bBytes = source.bytes("b");
	ASSERT_TRUE(aBytes.isOk());
	ASSERT_TRUE(bBytes.isOk());
	EXPECT_EQ((*aBytes)[0], 11u);
	EXPECT_EQ((*bBytes)[0], 22u);
	EXPECT_EQ(source.metadata().at("total_size"), "2");
}

TEST_F(TransferWeightsTest, RejectsUnindexedShardWeights) {
	const oa::U8 values[] = {1, 2};
	ASSERT_TRUE(writeSafeTensor(file("model-00001-of-00001.safetensors"),
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"hidden\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", values).isOk());
	const oa::String index =
		"{\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\"}}";
	ASSERT_TRUE(oa::Filesystem::writeText(file("model.safetensors.index.json"), index).isOk());
	auto result = oa::openWeightSource(directory_);
	ASSERT_TRUE(result.isError());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsDuplicateEntryNames) {
	const oa::String header =
		"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const oa::U8 payload[] = {1, 2};
	ASSERT_TRUE(writeSafeTensor(file("duplicate.safetensors"), header, payload).isOk());

	auto result = oa::openWeightSource(file("duplicate.safetensors"));
	ASSERT_TRUE(result.isError());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsOverlappingPayloads) {
	const oa::String header =
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
		"\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
	const oa::U8 payload[] = {0, 1, 2, 3, 4, 5};
	ASSERT_TRUE(writeSafeTensor(file("overlap.safetensors"), header, payload).isOk());

	auto result = oa::openWeightSource(file("overlap.safetensors"));
	ASSERT_TRUE(result.isError());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, AcceptsUnalignedMixedDtypePayloads) {
	const oa::String header =
		"{\"prefix\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,5]}}";
	const oa::U8 payload[] = {7, 0, 0, 128, 63};
	ASSERT_TRUE(writeSafeTensor(file("unaligned.safetensors"), header, payload).isOk());

	auto sourceResult = oa::openWeightSource(file("unaligned.safetensors"));
	ASSERT_TRUE(sourceResult.isOk());
	auto bytes = sourceResult.getValue()->bytes("x");
	ASSERT_TRUE(bytes.isOk());
	oa::F32 value = 0.0F;
	oa::memcpy(&value, bytes->data(), sizeof(value));
	EXPECT_FLOAT_EQ(value, 1.0F);
}

TEST_F(TransferWeightsTest, RejectsShapeByteMismatchAndOutOfBounds) {
	const oa::U8 payload[] = {0, 0, 0, 0};

	const oa::String mismatch =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}";
	ASSERT_TRUE(writeSafeTensor(file("mismatch.safetensors"), mismatch, payload).isOk());
	auto mismatchResult = oa::openWeightSource(file("mismatch.safetensors"));
	ASSERT_TRUE(mismatchResult.isError());
	EXPECT_EQ(mismatchResult.getStatus().getCode(), oa::StatusCode::FileCorrupt);

	const oa::String outside =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
	ASSERT_TRUE(writeSafeTensor(file("outside.safetensors"), outside, payload).isOk());
	auto outsideResult = oa::openWeightSource(file("outside.safetensors"));
	ASSERT_TRUE(outsideResult.isError());
	EXPECT_EQ(outsideResult.getStatus().getCode(), oa::StatusCode::FileCorrupt);

	const oa::String hole =
		"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const oa::U8 holePayload[] = {0, 1};
	ASSERT_TRUE(writeSafeTensor(file("hole.safetensors"), hole, holePayload).isOk());
	auto holeResult = oa::openWeightSource(file("hole.safetensors"));
	ASSERT_TRUE(holeResult.isError());
	EXPECT_EQ(holeResult.getStatus().getCode(), oa::StatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsElementCountOverflowAndUnknownDtype) {
	const oa::String overflow =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,3],\"data_offsets\":[0,0]}}";
	ASSERT_TRUE(writeSafeTensor(file("overflow.safetensors"), overflow, {}).isOk());
	auto overflowResult = oa::openWeightSource(file("overflow.safetensors"));
	ASSERT_TRUE(overflowResult.isError());
	EXPECT_EQ(overflowResult.getStatus().getCode(), oa::StatusCode::FileCorrupt);

	const oa::String unknown =
		"{\"x\":{\"dtype\":\"Q4\",\"shape\":[1],\"data_offsets\":[0,1]}}";
	const oa::U8 payload[] = {0};
	ASSERT_TRUE(writeSafeTensor(file("unknown.safetensors"), unknown, payload).isOk());
	auto unknownResult = oa::openWeightSource(file("unknown.safetensors"));
	ASSERT_TRUE(unknownResult.isError());
	EXPECT_EQ(unknownResult.getStatus().getCode(), oa::StatusCode::DtypeMismatch);
}
