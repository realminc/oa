#include "../../oaTest.h"

#include <ml/nn/alm/clipTextAg.h>
#include <ml/nn/alm/clipTokenizer.h>
#include <ml/nn/alm/almAg.h>

#include <cmath>
#include <fstream>
#include <vector>

namespace {

class ClipText : public ::testing::Test {};

oa::Matrix intMatrix(oa::Span<const oa::I32> inValues, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromInt32(inValues, inShape, oa::ScalarType::Int32);
}

std::vector<oa::U8> readBytes(const char* inPath) {
	std::ifstream file(inPath, std::ios::binary | std::ios::ate);
	if (not file.good()) return {};
	const auto size = file.tellg();
	if (size <= 0) return {};
	file.seekg(0);
	std::vector<oa::U8> bytes(static_cast<size_t>(size));
	file.read(reinterpret_cast<char*>(bytes.data()), size);
	if (not file.good()) return {};
	return bytes;
}

} // namespace

TEST_VK(ClipText, ScaledArchitectureIsFrozenAndPromptSensitive) {
	oa::ClipTextConfig cfg;
	cfg.vocabSize = 32;
	cfg.contextLength = 5;
	cfg.hiddenSize = 8;
	cfg.intermediateSize = 16;
	cfg.numHeads = 2;
	cfg.numLayers = 2;
	cfg.projectionDim = 6;
	cfg.bosToken = 30;
	cfg.eosToken = 31;
	cfg.padToken = 31;
	oa::ClipTextAg model(cfg);

	const auto parameters = model.allNamedParameterPtrs();
	EXPECT_EQ(parameters.size(), 37u);
	for (const auto& named : parameters) {
		EXPECT_FALSE(named.param->requiresGrad) << named.path.cStr();
		EXPECT_FALSE(named.param->data.requiresGrad()) << named.path.cStr();
	}

	const oa::I32 ids[] = {
		30, 1, 2, 31, 31,
		30, 7, 8, 9, 31,
	};
	const oa::I32 eosRows[] = {3, 9};
	auto output = model.forwardTokens(
		intMatrix(oa::Span<const oa::I32>(ids), oa::MatrixShape{2, 5}),
		intMatrix(oa::Span<const oa::I32>(eosRows), oa::MatrixShape{2}));
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	EXPECT_EQ(output.getShape(), (oa::MatrixShape{2, 6}));
	const auto* values = output.dataAs<const oa::F32>();
	oa::F64 delta = 0.0;
	for (oa::I64 i = 0; i < output.numElements(); ++i) EXPECT_TRUE(std::isfinite(values[i]));
	for (oa::I32 d = 0; d < cfg.projectionDim; ++d)
		delta += std::abs(values[d] - values[cfg.projectionDim + d]);
	EXPECT_GT(delta, 1e-6);
}

TEST_VK(ClipText, RejectsInvalidTopologyAndInputContract) {
	oa::ClipTextConfig bad;
	bad.hiddenSize = 10;
	bad.numHeads = 3;
	EXPECT_TRUE(bad.validate().isError());

	oa::ClipTextConfig cfg;
	cfg.vocabSize = 16;
	cfg.contextLength = 3;
	cfg.hiddenSize = 4;
	cfg.intermediateSize = 8;
	cfg.numHeads = 2;
	cfg.numLayers = 1;
	cfg.projectionDim = 4;
	cfg.bosToken = 14;
	cfg.eosToken = 15;
	cfg.padToken = 15;
	oa::ClipTextAg model(cfg);
	const oa::I32 ids[] = {14, 1, 15};
	const oa::I32 wrongRows[] = {2, 2};
	EXPECT_THROW((void)model.forwardTokens(
		intMatrix(oa::Span<const oa::I32>(ids), oa::MatrixShape{1, 3}),
		intMatrix(oa::Span<const oa::I32>(wrongRows), oa::MatrixShape{2})), std::invalid_argument);
}

TEST_VK(ClipText, TokenizerMatchesPinnedOpenAiIds) {
	const char* merges = std::getenv("OA_CLIP_MERGES");
	if (merges == nullptr or merges[0] == '\0')
		GTEST_SKIP() << "set OA_CLIP_MERGES to the pinned openai/clip-vit-large-patch14 merges.txt";
	oa::ClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.loadMerges(oa::Path(merges)).isOk());
	EXPECT_EQ(tokenizer.vocabSize(), 49408);
	EXPECT_EQ(tokenizer.bosToken(), 49406);
	EXPECT_EQ(tokenizer.eosToken(), 49407);

	const oa::String prompts[] = {
		"hello world",
		"a person walks forward, turns left, and raises both arms",
		"We're testing UTF-8 café — fast!",
		"",
	};
	auto encoded = tokenizer.encode(oa::Span<const oa::String>(prompts), 77, true);
	ASSERT_TRUE(encoded.isOk()) << encoded.getStatus().getMessage().cStr();
	const auto& ids = encoded.getValue().tokenIds;
	const oa::I32 expected0[] = {49406, 3306, 1002, 49407};
	const oa::I32 expected1[] = {49406, 320, 2533, 8192, 2342, 267, 3185, 1823, 267, 537, 13297, 2212, 5706, 49407};
	const oa::I32 expected2[] = {49406, 649, 982, 4967, 1419, 325, 268, 279, 15304, 2005, 1953, 256, 49407};
	const oa::I32 expected3[] = {49406, 49407};
	const oa::Span<const oa::I32> expected[] = {expected0, expected1, expected2, expected3};
	for (oa::I32 b = 0; b < 4; ++b) {
		for (oa::Usize i = 0; i < expected[b].size(); ++i)
			EXPECT_EQ(ids[static_cast<oa::Usize>(b) * 77 + i], expected[b][i]);
		EXPECT_EQ(encoded.getValue().flatEosRows[static_cast<oa::Usize>(b)],
			b * 77 + static_cast<oa::I32>(expected[b].size()) - 1);
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
		oa::U32 value = 0;
		file.read(reinterpret_cast<char*>(&value), sizeof(value));
		return value;
	};
	ASSERT_EQ(readU32(), 0x50494C43U); // "CLIP" little-endian
	ASSERT_EQ(readU32(), 1U);
	const oa::U32 count = readU32();
	ASSERT_GT(count, 0U);
	oa::Vec<oa::String> prompts;
	prompts.reserve(count);
	std::vector<oa::I32> expected(static_cast<size_t>(count) * 77);
	for (oa::U32 i = 0; i < count; ++i) {
		const oa::U32 length = readU32();
		std::string prompt(length, '\0');
		file.read(prompt.data(), static_cast<std::streamsize>(length));
		file.read(reinterpret_cast<char*>(expected.data() + static_cast<size_t>(i) * 77),
			77 * static_cast<std::streamsize>(sizeof(oa::I32)));
		ASSERT_TRUE(file.good());
		prompts.pushBack(oa::String(prompt.c_str(), prompt.size()));
	}
	oa::ClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.loadMerges(oa::Path(merges)).isOk());
	auto actual = tokenizer.encode(oa::Span<const oa::String>(prompts.data(), prompts.size()), 77, true);
	ASSERT_TRUE(actual.isOk()) << actual.getStatus().getMessage().cStr();
	ASSERT_EQ(actual.getValue().tokenIds.size(), expected.size());
	for (oa::Usize i = 0; i < expected.size(); ++i)
		ASSERT_EQ(actual.getValue().tokenIds[i], expected[i]) << "flat token index " << i;
	std::printf("CLIP tokenizer corpus parity: %u captions, %zu token IDs\n",
		count, expected.size());
}

TEST_VK(ClipText, FullCheckpointMatchesHuggingFaceProjection) {
	const char* merges = std::getenv("OA_CLIP_MERGES");
	const char* modelPath = std::getenv("OA_CLIP_OAM");
	const char* referencePath = std::getenv("OA_CLIP_REFERENCE");
	if (merges == nullptr or modelPath == nullptr or referencePath == nullptr)
		GTEST_SKIP() << "set OA_CLIP_MERGES, OA_CLIP_OAM and OA_CLIP_REFERENCE for full parity";

	oa::ClipTokenizer tokenizer;
	ASSERT_TRUE(tokenizer.loadMerges(oa::Path(merges)).isOk());
	const oa::String prompts[] = {
		"hello world",
		"a person walks forward, turns left, and raises both arms",
		"We're testing UTF-8 café — fast!",
		"",
	};
	auto encoded = tokenizer.encode(oa::Span<const oa::String>(prompts), 77, true);
	ASSERT_TRUE(encoded.isOk());
	auto loaded = oa::ClipTextAg::loadArchive(testEngine(), modelPath);
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().getMessage().cStr();

	const auto& batch = encoded.getValue();
	auto ids = intMatrix(oa::Span<const oa::I32>(batch.tokenIds.data(), batch.tokenIds.size()),
		oa::MatrixShape{batch.batch, batch.contextLength});
	auto eos = intMatrix(oa::Span<const oa::I32>(batch.flatEosRows.data(), batch.flatEosRows.size()),
		oa::MatrixShape{batch.batch});
	auto output = loaded.getValue()->forwardTokens(ids, eos);
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());

	std::ifstream refFile(referencePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(refFile.good());
	const auto bytes = refFile.tellg();
	ASSERT_EQ(bytes, static_cast<std::streamoff>(output.numElements() * sizeof(oa::F32)));
	refFile.seekg(0);
	std::vector<oa::F32> reference(static_cast<size_t>(output.numElements()));
	refFile.read(reinterpret_cast<char*>(reference.data()), bytes);
	ASSERT_TRUE(refFile.good());
	const auto* actual = output.dataAs<const oa::F32>();
	oa::F32 maxError = 0.0F;
	oa::F64 squaredError = 0.0;
	for (oa::I64 i = 0; i < output.numElements(); ++i) {
		const oa::F32 error = std::abs(actual[i] - reference[static_cast<size_t>(i)]);
		maxError = std::max(maxError, error);
		squaredError += static_cast<oa::F64>(error) * error;
	}
	const oa::F64 rmse = std::sqrt(squaredError / static_cast<oa::F64>(output.numElements()));
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

	auto merges = readBytes(mergesPath);
	ASSERT_FALSE(merges.empty());
	auto clipResult = oa::ClipTextAg::loadArchive(testEngine(), modelPath);
	ASSERT_TRUE(clipResult.isOk()) << clipResult.getStatus().getMessage().cStr();
	auto clip = std::move(clipResult).getValue();

	oa::AlmTokenizerConfig tokenizerConfig;
	tokenizerConfig.inputDim = 6;
	tokenizerConfig.width = 8;
	tokenizerConfig.codeDim = 8;
	tokenizerConfig.numCodes = 8;
	tokenizerConfig.downT = 1;
	tokenizerConfig.depth = 1;
	oa::AlmPriorConfig priorConfig;
	priorConfig.syncVocab(tokenizerConfig.numCodes);
	priorConfig.dModel = 8;
	priorConfig.numHeads = 2;
	priorConfig.numLayers = 1;
	priorConfig.dFfn = 16;
	priorConfig.textFeatureDim = oa::ClipTextConfig::viTL14().projectionDim;
	priorConfig.seqLen = 4;
	priorConfig.maxSeqLen = 8;
	priorConfig.maxGenLen = 7;
	auto alm = oa::makeShared<oa::AlmAg>(
		oa::makeShared<oa::AlmTokenizerAg>(tokenizerConfig),
		oa::makeShared<oa::AlmPriorAg>(priorConfig), clip,
		oa::Span<const oa::U8>(merges.data(), merges.size()),
		"openai/clip-vit-large-patch14");

	auto beforeResult = alm->encodePrompt("hello world");
	ASSERT_TRUE(beforeResult.isOk()) << beforeResult.getStatus().getMessage().cStr();
	auto before = beforeResult.getValue();
	auto& ctx = oa::ExecutionSession::getActive();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	std::vector<oa::F32> expected(static_cast<size_t>(before.numElements()));
	{
		std::ifstream reference(referencePath, std::ios::binary);
		ASSERT_TRUE(reference.good());
		reference.read(reinterpret_cast<char*>(expected.data()),
			static_cast<std::streamsize>(expected.size() * sizeof(oa::F32)));
		ASSERT_TRUE(reference.good());
	}
	const auto* beforeValues = before.dataAs<const oa::F32>();
	for (oa::Usize i = 0; i < expected.size(); ++i)
		EXPECT_NEAR(beforeValues[i], expected[i], 2e-3F);

	const oa::String bundlePath = "/tmp/oa_alm_native_clip_bundle.oam";
	ASSERT_TRUE(alm->saveBundle(testEngine(), bundlePath).isOk());
	auto bundle = oa::ModelFile::load(bundlePath);
	ASSERT_TRUE(bundle.isOk());
	EXPECT_NE(bundle.getValue().findWeight("text_encoder.text_projection.weight"), nullptr);
	EXPECT_NE(bundle.getValue().findState("text_tokenizer_merges"), nullptr);
	auto loadedResult = oa::AlmAg::loadBundle(testEngine(), bundlePath);
	ASSERT_TRUE(loadedResult.isOk()) << loadedResult.getStatus().getMessage().cStr();
	auto loaded = std::move(loadedResult).getValue();
	ASSERT_TRUE(loaded->hasNativeTextEncoder());
	auto afterResult = loaded->encodePrompt("hello world");
	ASSERT_TRUE(afterResult.isOk()) << afterResult.getStatus().getMessage().cStr();
	auto after = afterResult.getValue();
	ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
	ASSERT_EQ(before.numElements(), after.numElements());
	const auto* afterValues = after.dataAs<const oa::F32>();
	for (oa::I64 i = 0; i < before.numElements(); ++i)
		EXPECT_EQ(beforeValues[i], afterValues[i]);
	std::remove(bundlePath.cStr());
}
