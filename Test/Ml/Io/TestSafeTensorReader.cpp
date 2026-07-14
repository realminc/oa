#include "../../OaTest.h"

#include <Oa/Ml/TransferWeights.h>

#include <atomic>
#include <chrono>

namespace {

std::atomic<OaU64> GSafeTensorTestSequence{0};

OaPath MakeTestDirectory() {
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	const OaString name = OaString("oa_safetensor_") +
		std::to_string(static_cast<unsigned long long>(++GSafeTensorTestSequence)) + "_" +
		std::to_string(static_cast<long long>(tick));
	return OaFileIo::GetTempDirectory() / OaPath(name);
}

OaStatus WriteSafeTensor(
	const OaPath& InPath,
	OaStringView InHeader,
	OaSpan<const OaU8> InPayload
) {
	const OaU64 headerSize = static_cast<OaU64>(InHeader.size());
	OaVec<OaU8> bytes(static_cast<OaUsize>(8 + headerSize + InPayload.Size()));
	OaMemcpy(bytes.Data(), &headerSize, sizeof(headerSize));
	OaMemcpy(bytes.Data() + 8, InHeader.data(), static_cast<OaUsize>(headerSize));
	if (!InPayload.Empty()) {
		OaMemcpy(bytes.Data() + 8 + headerSize, InPayload.Data(), InPayload.Size());
	}
	return OaFileIo::WriteBinary(InPath, {bytes.Data(), bytes.Size()});
}

class TransferWeightsTest : public ::testing::Test {
protected:
	void SetUp() override {
		Directory_ = MakeTestDirectory();
		ASSERT_TRUE(OaFileIo::CreateDirectories(Directory_).IsOk());
	}

	void TearDown() override {
		(void)OaFileIo::RemoveDirectory(Directory_, true);
	}

	OaPath File(const char* InName) const {
		return Directory_ / OaPath(InName);
	}

	OaPath Directory_;
};

} // namespace

TEST_F(TransferWeightsTest, OpensAndExposesCheckedEntrySpans) {
	const OaString header =
		"{\"__metadata__\":{\"format\":\"pt\"},"
		"\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
	const OaF32 values[] = {1.0F, 2.0F, 3.0F, 4.0F};
	const auto payload = OaSpan<const OaU8>(
		reinterpret_cast<const OaU8*>(values), sizeof(values));
	ASSERT_TRUE(WriteSafeTensor(File("valid.safetensors"), header, payload).IsOk());

	auto sourceResult = OaOpenWeightSource(File("valid.safetensors"));
	ASSERT_TRUE(sourceResult.IsOk()) << sourceResult.GetStatus().ToString().c_str();
	auto& source = *sourceResult.GetValue();
	EXPECT_EQ(source.List().Size(), 1u);

	const auto* entry = source.Find("weight");
	ASSERT_NE(entry, nullptr);
	EXPECT_EQ(entry->Dtype, OaScalarType::Float32);
	EXPECT_EQ(entry->ElementCount, 4u);
	EXPECT_EQ(entry->ByteSize, sizeof(values));

	auto bytes = source.Bytes("weight");
	ASSERT_TRUE(bytes.IsOk()) << bytes.GetStatus().ToString().c_str();
	ASSERT_EQ(bytes->Size(), sizeof(values));
	EXPECT_EQ(OaMemcmp(bytes->Data(), values, sizeof(values)), 0);
	EXPECT_EQ(source.Metadata().At("format"), "pt");
}

TEST_F(TransferWeightsTest, ConvertsFloat32ToBfloat16) {
	const OaString header =
		"{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
	const OaF32 values[] = {1.0F, -2.0F};
	ASSERT_TRUE(WriteSafeTensor(
		File("convert.safetensors"), header,
		{reinterpret_cast<const OaU8*>(values), sizeof(values)}).IsOk());

	auto sourceResult = OaOpenWeightSource(File("convert.safetensors"));
	ASSERT_TRUE(sourceResult.IsOk());
	auto& source = *sourceResult.GetValue();
	OaU16 converted[2] = {};
	ASSERT_TRUE(source.Read(
		"weight",
		{reinterpret_cast<OaU8*>(converted), sizeof(converted)},
		OaScalarType::BFloat16).IsOk());
	EXPECT_EQ(converted[0], 0x3F80u);
	EXPECT_EQ(converted[1], 0xC000u);
}

TEST_F(TransferWeightsTest, ExecutesIdentityTransposeConcatAndSliceMappings) {
	const OaString header =
		"{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
		"\"b\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[16,24]}}";
	const OaF32 values[] = {1, 2, 3, 4, 5, 6};
	ASSERT_TRUE(WriteSafeTensor(File("mapped.safetensors"), header,
		{reinterpret_cast<const OaU8*>(values), sizeof(values)}).IsOk());
	auto sourceResult = OaOpenWeightSource(File("mapped.safetensors"));
	ASSERT_TRUE(sourceResult.IsOk());

	OaWeightMap map;
	map.Architecture = "mapping-test";
	map.Mappings = {
		{.Sources = {"a"}, .Target = "identity", .TargetShape = {2, 2},
			.TargetDtype = OaScalarType::Float32, .Transform = OaWeightTransform::Identity},
		{.Sources = {"a"}, .Target = "transpose", .TargetShape = {2, 2},
			.TargetDtype = OaScalarType::Float32, .Transform = OaWeightTransform::Transpose2D},
		{.Sources = {"a", "b"}, .Target = "concat", .TargetShape = {2, 3},
			.TargetDtype = OaScalarType::Float32, .Transform = OaWeightTransform::Concat,
			.ConcatAxis = 1},
		{.Sources = {"a"}, .Target = "slice", .TargetShape = {2, 1},
			.TargetDtype = OaScalarType::Float32, .Transform = OaWeightTransform::Slice,
			.Slice = {.Axis = 1, .Begin = 1, .Length = 1}},
	};
	OamModel model;
	auto reportResult = OaTransferWeights(*sourceResult.GetValue(), map, model);
	ASSERT_TRUE(reportResult.IsOk()) << reportResult.GetStatus().ToString().c_str();
	EXPECT_EQ(reportResult->SourceWeights, 2u);
	EXPECT_EQ(reportResult->UsedSourceWeights, 2u);
	EXPECT_EQ(reportResult->OutputWeights, 4u);
	EXPECT_TRUE(reportResult->UnusedSources.Empty());

	const OaF32 identity[] = {1, 2, 3, 4};
	const OaF32 transpose[] = {1, 3, 2, 4};
	const OaF32 concat[] = {1, 2, 5, 3, 4, 6};
	const OaF32 slice[] = {2, 4};
	ASSERT_NE(model.WeightPtr("identity"), nullptr);
	EXPECT_EQ(OaMemcmp(model.WeightPtr("identity"), identity, sizeof(identity)), 0);
	EXPECT_EQ(OaMemcmp(model.WeightPtr("transpose"), transpose, sizeof(transpose)), 0);
	EXPECT_EQ(OaMemcmp(model.WeightPtr("concat"), concat, sizeof(concat)), 0);
	EXPECT_EQ(OaMemcmp(model.WeightPtr("slice"), slice, sizeof(slice)), 0);
}

TEST_F(TransferWeightsTest, EnforcesCompleteSourceCoverage) {
	const OaString header =
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"unused\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const OaU8 values[] = {1, 2};
	ASSERT_TRUE(WriteSafeTensor(File("coverage.safetensors"), header, values).IsOk());
	auto sourceResult = OaOpenWeightSource(File("coverage.safetensors"));
	ASSERT_TRUE(sourceResult.IsOk());
	OaWeightMap map;
	map.Architecture = "coverage-test";
	map.Mappings.PushBack({.Sources = {"a"}, .Target = "a", .TargetShape = {1},
		.TargetDtype = OaScalarType::UInt8});
	OamModel model;
	auto strictResult = OaTransferWeights(*sourceResult.GetValue(), map, model);
	ASSERT_TRUE(strictResult.IsError());
	EXPECT_EQ(strictResult.GetStatus().GetCode(), OaStatusCode::FailedPrecondition);

	map.RequireAllSourceWeights = false;
	auto permissiveResult = OaTransferWeights(*sourceResult.GetValue(), map, model);
	ASSERT_TRUE(permissiveResult.IsOk());
	ASSERT_EQ(permissiveResult->UnusedSources.Size(), 1u);
	EXPECT_EQ(permissiveResult->UnusedSources[0], "unused");
}

TEST_F(TransferWeightsTest, OpensIndexedShardPackageAsOneWeightSource) {
	const OaU8 a[] = {11};
	const OaU8 b[] = {22};
	ASSERT_TRUE(WriteSafeTensor(File("model-00001-of-00002.safetensors"),
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", a).IsOk());
	ASSERT_TRUE(WriteSafeTensor(File("model-00002-of-00002.safetensors"),
		"{\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", b).IsOk());
	const OaString index =
		"{\"metadata\":{\"total_size\":\"2\"},\"weight_map\":{"
		"\"a\":\"model-00001-of-00002.safetensors\","
		"\"b\":\"model-00002-of-00002.safetensors\"}}";
	ASSERT_TRUE(OaFileIo::WriteText(File("model.safetensors.index.json"), index).IsOk());

	auto sourceResult = OaOpenWeightSource(Directory_);
	ASSERT_TRUE(sourceResult.IsOk()) << sourceResult.GetStatus().ToString().c_str();
	auto& source = *sourceResult.GetValue();
	ASSERT_EQ(source.List().Size(), 2u);
	ASSERT_NE(source.Find("a"), nullptr);
	ASSERT_NE(source.Find("b"), nullptr);
	auto aBytes = source.Bytes("a");
	auto bBytes = source.Bytes("b");
	ASSERT_TRUE(aBytes.IsOk());
	ASSERT_TRUE(bBytes.IsOk());
	EXPECT_EQ((*aBytes)[0], 11u);
	EXPECT_EQ((*bBytes)[0], 22u);
	EXPECT_EQ(source.Metadata().At("total_size"), "2");
}

TEST_F(TransferWeightsTest, RejectsUnindexedShardWeights) {
	const OaU8 values[] = {1, 2};
	ASSERT_TRUE(WriteSafeTensor(File("model-00001-of-00001.safetensors"),
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"hidden\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", values).IsOk());
	const OaString index =
		"{\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\"}}";
	ASSERT_TRUE(OaFileIo::WriteText(File("model.safetensors.index.json"), index).IsOk());
	auto result = OaOpenWeightSource(Directory_);
	ASSERT_TRUE(result.IsError());
	EXPECT_EQ(result.GetStatus().GetCode(), OaStatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsDuplicateEntryNames) {
	const OaString header =
		"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const OaU8 payload[] = {1, 2};
	ASSERT_TRUE(WriteSafeTensor(File("duplicate.safetensors"), header, payload).IsOk());

	auto result = OaOpenWeightSource(File("duplicate.safetensors"));
	ASSERT_TRUE(result.IsError());
	EXPECT_EQ(result.GetStatus().GetCode(), OaStatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsOverlappingPayloads) {
	const OaString header =
		"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
		"\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[2,6]}}";
	const OaU8 payload[] = {0, 1, 2, 3, 4, 5};
	ASSERT_TRUE(WriteSafeTensor(File("overlap.safetensors"), header, payload).IsOk());

	auto result = OaOpenWeightSource(File("overlap.safetensors"));
	ASSERT_TRUE(result.IsError());
	EXPECT_EQ(result.GetStatus().GetCode(), OaStatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, AcceptsUnalignedMixedDtypePayloads) {
	const OaString header =
		"{\"prefix\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
		"\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,5]}}";
	const OaU8 payload[] = {7, 0, 0, 128, 63};
	ASSERT_TRUE(WriteSafeTensor(File("unaligned.safetensors"), header, payload).IsOk());

	auto sourceResult = OaOpenWeightSource(File("unaligned.safetensors"));
	ASSERT_TRUE(sourceResult.IsOk());
	auto bytes = sourceResult.GetValue()->Bytes("x");
	ASSERT_TRUE(bytes.IsOk());
	OaF32 value = 0.0F;
	OaMemcpy(&value, bytes->Data(), sizeof(value));
	EXPECT_FLOAT_EQ(value, 1.0F);
}

TEST_F(TransferWeightsTest, RejectsShapeByteMismatchAndOutOfBounds) {
	const OaU8 payload[] = {0, 0, 0, 0};

	const OaString mismatch =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}";
	ASSERT_TRUE(WriteSafeTensor(File("mismatch.safetensors"), mismatch, payload).IsOk());
	auto mismatchResult = OaOpenWeightSource(File("mismatch.safetensors"));
	ASSERT_TRUE(mismatchResult.IsError());
	EXPECT_EQ(mismatchResult.GetStatus().GetCode(), OaStatusCode::FileCorrupt);

	const OaString outside =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
	ASSERT_TRUE(WriteSafeTensor(File("outside.safetensors"), outside, payload).IsOk());
	auto outsideResult = OaOpenWeightSource(File("outside.safetensors"));
	ASSERT_TRUE(outsideResult.IsError());
	EXPECT_EQ(outsideResult.GetStatus().GetCode(), OaStatusCode::FileCorrupt);

	const OaString hole =
		"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
	const OaU8 holePayload[] = {0, 1};
	ASSERT_TRUE(WriteSafeTensor(File("hole.safetensors"), hole, holePayload).IsOk());
	auto holeResult = OaOpenWeightSource(File("hole.safetensors"));
	ASSERT_TRUE(holeResult.IsError());
	EXPECT_EQ(holeResult.GetStatus().GetCode(), OaStatusCode::FileCorrupt);
}

TEST_F(TransferWeightsTest, RejectsElementCountOverflowAndUnknownDtype) {
	const OaString overflow =
		"{\"x\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,3],\"data_offsets\":[0,0]}}";
	ASSERT_TRUE(WriteSafeTensor(File("overflow.safetensors"), overflow, {}).IsOk());
	auto overflowResult = OaOpenWeightSource(File("overflow.safetensors"));
	ASSERT_TRUE(overflowResult.IsError());
	EXPECT_EQ(overflowResult.GetStatus().GetCode(), OaStatusCode::FileCorrupt);

	const OaString unknown =
		"{\"x\":{\"dtype\":\"Q4\",\"shape\":[1],\"data_offsets\":[0,1]}}";
	const OaU8 payload[] = {0};
	ASSERT_TRUE(WriteSafeTensor(File("unknown.safetensors"), unknown, payload).IsOk());
	auto unknownResult = OaOpenWeightSource(File("unknown.safetensors"));
	ASSERT_TRUE(unknownResult.IsError());
	EXPECT_EQ(unknownResult.GetStatus().GetCode(), OaStatusCode::DtypeMismatch);
}
