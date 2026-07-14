#include "../../OaTest.h"

#include <Ml/Nn/Alm/ClipTextAg.h>
#include <Ml/Nn/Alm/ClipTokenizer.h>
#include <Ml/Nn/Alm/AlmAg.h>

#include <cmath>
#include <fstream>
#include <vector>

namespace {

class ClipText : public ::testing::Test {};

OaMatrix IntMatrix(OaSpan<const OaI32> InValues, OaMatrixShape InShape) {
	return OaFnMatrix::FromInt32(InValues, InShape, OaScalarType::Int32);
}

std::vector<OaU8> ReadBytes(const char* InPath) {
	std::ifstream file(InPath, std::ios::binary | std::ios::ate);
	if (not file.good()) return {};
	const auto size = file.tellg();
	if (size <= 0) return {};
	file.seekg(0);
	std::vector<OaU8> bytes(static_cast<size_t>(size));
	file.read(reinterpret_cast<char*>(bytes.data()), size);
	if (not file.good()) return {};
	return bytes;
}

} // namespace

TEST_VK(ClipText, ScaledArchitectureIsFrozenAndPromptSensitive) {
	OaClipTextConfig cfg;
	cfg.VocabSize = 32;
	cfg.ContextLength = 5;
	cfg.HiddenSize = 8;
	cfg.IntermediateSize = 16;
	cfg.NumHeads = 2;
	cfg.NumLayers = 2;
	cfg.ProjectionDim = 6;
	cfg.BosToken = 30;
	cfg.EosToken = 31;
	cfg.PadToken = 31;
	OaClipTextAg model(cfg);

	const auto parameters = model.AllNamedParameterPtrs();
	EXPECT_EQ(parameters.Size(), 37u);
	for (const auto& named : parameters) {
		EXPECT_FALSE(named.Param->RequiresGrad) << named.Path.CStr();
		EXPECT_FALSE(named.Param->Data.RequiresGrad()) << named.Path.CStr();
	}

	const OaI32 ids[] = {
		30, 1, 2, 31, 31,
		30, 7, 8, 9, 31,
	};
	const OaI32 eosRows[] = {3, 9};
	auto output = model.ForwardTokens(
		IntMatrix(OaSpan<const OaI32>(ids), OaMatrixShape{2, 5}),
		IntMatrix(OaSpan<const OaI32>(eosRows), OaMatrixShape{2}));
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	EXPECT_EQ(output.GetShape(), (OaMatrixShape{2, 6}));
	const auto* values = output.DataAs<const OaF32>();
	OaF64 delta = 0.0;
	for (OaI64 i = 0; i < output.NumElements(); ++i) EXPECT_TRUE(std::isfinite(values[i]));
	for (OaI32 d = 0; d < cfg.ProjectionDim; ++d)
		delta += std::abs(values[d] - values[cfg.ProjectionDim + d]);
	EXPECT_GT(delta, 1e-6);
}

TEST_VK(ClipText, RejectsInvalidTopologyAndInputContract) {
	OaClipTextConfig bad;
	bad.HiddenSize = 10;
	bad.NumHeads = 3;
	EXPECT_TRUE(bad.Validate().IsError());

	OaClipTextConfig cfg;
	cfg.VocabSize = 16;
	cfg.ContextLength = 3;
	cfg.HiddenSize = 4;
	cfg.IntermediateSize = 8;
	cfg.NumHeads = 2;
	cfg.NumLayers = 1;
	cfg.ProjectionDim = 4;
	cfg.BosToken = 14;
	cfg.EosToken = 15;
	cfg.PadToken = 15;
	OaClipTextAg model(cfg);
	const OaI32 ids[] = {14, 1, 15};
	const OaI32 wrongRows[] = {2, 2};
	EXPECT_THROW((void)model.ForwardTokens(
		IntMatrix(OaSpan<const OaI32>(ids), OaMatrixShape{1, 3}),
		IntMatrix(OaSpan<const OaI32>(wrongRows), OaMatrixShape{2})), std::invalid_argument);
}

TEST_VK(ClipText, TokenizerMatchesPinnedOpenAiIds) {
	const char* merges = std::getenv("OA_CLIP_MERGES");
	if (merges == nullptr or merges[0] == '\0')
		GTEST_SKIP() << "set OA_CLIP_MERGES to the pinned openai/clip-vit-large-patch14 merges.txt";
	OaClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.LoadMerges(OaPath(merges)).IsOk());
	EXPECT_EQ(tokenizer.VocabSize(), 49408);
	EXPECT_EQ(tokenizer.BosToken(), 49406);
	EXPECT_EQ(tokenizer.EosToken(), 49407);

	const OaString prompts[] = {
		"hello world",
		"a person walks forward, turns left, and raises both arms",
		"We're testing UTF-8 café — fast!",
		"",
	};
	auto encoded = tokenizer.Encode(OaSpan<const OaString>(prompts), 77, true);
	ASSERT_TRUE(encoded.IsOk()) << encoded.GetStatus().GetMessage().CStr();
	const auto& ids = encoded.GetValue().TokenIds;
	const OaI32 expected0[] = {49406, 3306, 1002, 49407};
	const OaI32 expected1[] = {49406, 320, 2533, 8192, 2342, 267, 3185, 1823, 267, 537, 13297, 2212, 5706, 49407};
	const OaI32 expected2[] = {49406, 649, 982, 4967, 1419, 325, 268, 279, 15304, 2005, 1953, 256, 49407};
	const OaI32 expected3[] = {49406, 49407};
	const OaSpan<const OaI32> expected[] = {expected0, expected1, expected2, expected3};
	for (OaI32 b = 0; b < 4; ++b) {
		for (OaUsize i = 0; i < expected[b].Size(); ++i)
			EXPECT_EQ(ids[static_cast<OaUsize>(b) * 77 + i], expected[b][i]);
		EXPECT_EQ(encoded.GetValue().FlatEosRows[static_cast<OaUsize>(b)],
			b * 77 + static_cast<OaI32>(expected[b].Size()) - 1);
	}
}

TEST_VK(ClipText, CorpusTokenizerMatchesPinnedIds) {
	const char* merges = std::getenv("OA_CLIP_MERGES");
	const char* fixture = std::getenv("OA_CLIP_CORPUS_REFERENCE");
	if (merges == nullptr or fixture == nullptr)
		GTEST_SKIP() << "set OA_CLIP_MERGES and OA_CLIP_CORPUS_REFERENCE for corpus parity";
	std::ifstream file(fixture, std::ios::binary);
	ASSERT_TRUE(file.good());
	auto readU32 = [&]() {
		OaU32 value = 0;
		file.read(reinterpret_cast<char*>(&value), sizeof(value));
		return value;
	};
	ASSERT_EQ(readU32(), 0x50494C43U); // "CLIP" little-endian
	ASSERT_EQ(readU32(), 1U);
	const OaU32 count = readU32();
	ASSERT_GT(count, 0U);
	OaVec<OaString> prompts;
	prompts.Reserve(count);
	std::vector<OaI32> expected(static_cast<size_t>(count) * 77);
	for (OaU32 i = 0; i < count; ++i) {
		const OaU32 length = readU32();
		std::string prompt(length, '\0');
		file.read(prompt.data(), static_cast<std::streamsize>(length));
		file.read(reinterpret_cast<char*>(expected.data() + static_cast<size_t>(i) * 77),
			77 * static_cast<std::streamsize>(sizeof(OaI32)));
		ASSERT_TRUE(file.good());
		prompts.PushBack(OaString(prompt.c_str(), prompt.size()));
	}
	OaClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.LoadMerges(OaPath(merges)).IsOk());
	auto actual = tokenizer.Encode(OaSpan<const OaString>(prompts.Data(), prompts.Size()), 77, true);
	ASSERT_TRUE(actual.IsOk()) << actual.GetStatus().GetMessage().CStr();
	ASSERT_EQ(actual.GetValue().TokenIds.Size(), expected.size());
	for (OaUsize i = 0; i < expected.size(); ++i)
		ASSERT_EQ(actual.GetValue().TokenIds[i], expected[i]) << "flat token index " << i;
	std::printf("CLIP tokenizer corpus parity: %u captions, %zu token IDs\n",
		count, expected.size());
}

TEST_VK(ClipText, FullCheckpointMatchesHuggingFaceProjection) {
	const char* merges = std::getenv("OA_CLIP_MERGES");
	const char* modelPath = std::getenv("OA_CLIP_OAM");
	const char* referencePath = std::getenv("OA_CLIP_REFERENCE");
	if (merges == nullptr or modelPath == nullptr or referencePath == nullptr)
		GTEST_SKIP() << "set OA_CLIP_MERGES, OA_CLIP_OAM and OA_CLIP_REFERENCE for full parity";

	OaClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.LoadMerges(OaPath(merges)).IsOk());
	const OaString prompts[] = {
		"hello world",
		"a person walks forward, turns left, and raises both arms",
		"We're testing UTF-8 café — fast!",
		"",
	};
	auto encoded = tokenizer.Encode(OaSpan<const OaString>(prompts), 77, true);
	ASSERT_TRUE(encoded.IsOk());
	auto loaded = OaClipTextAg::LoadOam(modelPath);
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().GetMessage().CStr();

	const auto& batch = encoded.GetValue();
	auto ids = IntMatrix(OaSpan<const OaI32>(batch.TokenIds.Data(), batch.TokenIds.Size()),
		OaMatrixShape{batch.Batch, batch.ContextLength});
	auto eos = IntMatrix(OaSpan<const OaI32>(batch.FlatEosRows.Data(), batch.FlatEosRows.Size()),
		OaMatrixShape{batch.Batch});
	auto output = loaded.GetValue()->ForwardTokens(ids, eos);
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());

	std::ifstream refFile(referencePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(refFile.good());
	const auto bytes = refFile.tellg();
	ASSERT_EQ(bytes, static_cast<std::streamoff>(output.NumElements() * sizeof(OaF32)));
	refFile.seekg(0);
	std::vector<OaF32> reference(static_cast<size_t>(output.NumElements()));
	refFile.read(reinterpret_cast<char*>(reference.data()), bytes);
	ASSERT_TRUE(refFile.good());
	const auto* actual = output.DataAs<const OaF32>();
	OaF32 maxError = 0.0F;
	OaF64 squaredError = 0.0;
	for (OaI64 i = 0; i < output.NumElements(); ++i) {
		const OaF32 error = std::abs(actual[i] - reference[static_cast<size_t>(i)]);
		maxError = std::max(maxError, error);
		squaredError += static_cast<OaF64>(error) * error;
	}
	const OaF64 rmse = std::sqrt(squaredError / static_cast<OaF64>(output.NumElements()));
	std::printf("CLIP ViT-L/14 parity: max=%.8g rmse=%.8g\n",
		static_cast<double>(maxError), static_cast<double>(rmse));
	EXPECT_LT(maxError, 2e-3F);
	EXPECT_LT(rmse, 2e-4);
}

TEST_VK(ClipText, NativeAlmBundlePreservesPromptProjection) {
	const char* mergesPath = std::getenv("OA_CLIP_MERGES");
	const char* modelPath = std::getenv("OA_CLIP_OAM");
	const char* referencePath = std::getenv("OA_CLIP_REFERENCE");
	if (mergesPath == nullptr or modelPath == nullptr or referencePath == nullptr)
		GTEST_SKIP() << "set OA_CLIP_MERGES, OA_CLIP_OAM and OA_CLIP_REFERENCE for bundle parity";

	auto merges = ReadBytes(mergesPath);
	ASSERT_FALSE(merges.empty());
	auto clipResult = OaClipTextAg::LoadOam(modelPath);
	ASSERT_TRUE(clipResult.IsOk()) << clipResult.GetStatus().GetMessage().CStr();
	auto clip = std::move(clipResult).GetValue();

	OaAlmTokenizerConfig tokenizerConfig;
	tokenizerConfig.InputDim = 6;
	tokenizerConfig.Width = 8;
	tokenizerConfig.CodeDim = 8;
	tokenizerConfig.NumCodes = 8;
	tokenizerConfig.DownT = 1;
	tokenizerConfig.Depth = 1;
	OaAlmPriorConfig priorConfig;
	priorConfig.SyncVocab(tokenizerConfig.NumCodes);
	priorConfig.DModel = 8;
	priorConfig.NumHeads = 2;
	priorConfig.NumLayers = 1;
	priorConfig.DFfn = 16;
	priorConfig.TextFeatureDim = OaClipTextConfig::ViTL14().ProjectionDim;
	priorConfig.SeqLen = 4;
	priorConfig.MaxSeqLen = 8;
	priorConfig.MaxGenLen = 7;
	auto alm = OaMakeSharedPtr<OaAlmAg>(
		OaMakeSharedPtr<OaAlmTokenizerAg>(tokenizerConfig),
		OaMakeSharedPtr<OaAlmPriorAg>(priorConfig), clip,
		OaSpan<const OaU8>(merges.data(), merges.size()),
		"openai/clip-vit-large-patch14");

	auto beforeResult = alm->EncodePrompt("hello world");
	ASSERT_TRUE(beforeResult.IsOk()) << beforeResult.GetStatus().GetMessage().CStr();
	auto before = beforeResult.GetValue();
	auto& ctx = OaContext::GetDefault();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	std::vector<OaF32> expected(static_cast<size_t>(before.NumElements()));
	{
		std::ifstream reference(referencePath, std::ios::binary);
		ASSERT_TRUE(reference.good());
		reference.read(reinterpret_cast<char*>(expected.data()),
			static_cast<std::streamsize>(expected.size() * sizeof(OaF32)));
		ASSERT_TRUE(reference.good());
	}
	const auto* beforeValues = before.DataAs<const OaF32>();
	for (OaUsize i = 0; i < expected.size(); ++i)
		EXPECT_NEAR(beforeValues[i], expected[i], 2e-3F);

	const OaString bundlePath = "/tmp/oa_alm_native_clip_bundle.oam";
	ASSERT_TRUE(alm->SaveBundle(bundlePath).IsOk());
	auto bundle = OamModel::Load(bundlePath);
	ASSERT_TRUE(bundle.IsOk());
	EXPECT_NE(bundle.GetValue().FindWeight("text_encoder.text_projection.weight"), nullptr);
	EXPECT_NE(bundle.GetValue().FindState("text_tokenizer_merges"), nullptr);
	auto loadedResult = OaAlmAg::LoadBundle(bundlePath);
	ASSERT_TRUE(loadedResult.IsOk()) << loadedResult.GetStatus().GetMessage().CStr();
	auto loaded = std::move(loadedResult).GetValue();
	ASSERT_TRUE(loaded->HasNativeTextEncoder());
	auto afterResult = loaded->EncodePrompt("hello world");
	ASSERT_TRUE(afterResult.IsOk()) << afterResult.GetStatus().GetMessage().CStr();
	auto after = afterResult.GetValue();
	ASSERT_TRUE(ctx.Execute().IsOk());
	ASSERT_TRUE(ctx.Sync().IsOk());
	ASSERT_EQ(before.NumElements(), after.NumElements());
	const auto* afterValues = after.DataAs<const OaF32>();
	for (OaI64 i = 0; i < before.NumElements(); ++i)
		EXPECT_EQ(beforeValues[i], afterValues[i]);
	std::remove(bundlePath.CStr());
}
