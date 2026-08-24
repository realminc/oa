#include "../../oaTest.h"

#include <oa/ml/checkpoint.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/module.h>
#include <oa/ml/optim.h>

#include <chrono>
#include <cstring>
#include <type_traits>

namespace {

oa::Path makeDirectory() {
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	return oa::Paths::temp() /
		   oa::Path("oa_oam_integrity_" + std::to_string(static_cast<long long>(tick)));
}

oa::ModelFile makeModel(oa::F32 inLastValue = 4.0F) {
	oa::ModelFile model;
	std::strncpy(model.config.architecture, "IntegrityTest", sizeof(model.config.architecture) - 1);
	model.config.dModel = 4;
	model.progress.step = 17;
	const oa::F32 weights[] = {1.0F, 2.0F, 3.0F, inLastValue};
	const oa::U64 shape[] = {4};
	model.addWeight("linear.weight", oa::ScalarType::Float32, {shape, 1}, weights, sizeof(weights));
	return model;
}

oa::U64 legacyFileChecksum(const oa::Vec<oa::U8>& inBytes) {
	const auto* header = reinterpret_cast<const oa::ModelFileHeader*>(inBytes.data());
	const auto* sections = reinterpret_cast<const oa::ModelFileSectionHeader*>(
		inBytes.data() + sizeof(oa::ModelFileHeader));
	oa::U64 checksum = 0;
	for (oa::U32 i = 0; i < header->numSections; ++i)
		checksum ^= sections[i].checksum;
	return checksum;
}

void hashUpdate(oa::U64& inOutHash, const oa::U8* inData, oa::Usize inSize) {
	for (oa::Usize i = 0; i < inSize; ++i) {
		inOutHash ^= inData[i];
		inOutHash *= 0x100000001b3ULL;
	}
}

oa::U64 manifestChecksum(const oa::Vec<oa::U8>& inBytes) {
	auto header = *reinterpret_cast<const oa::ModelFileHeader*>(inBytes.data());
	header.checksum = 0;
	const auto* sections = inBytes.data() + sizeof(oa::ModelFileHeader);
	oa::U64 hash = 0xcbf29ce484222325ULL;
	hashUpdate(hash, reinterpret_cast<const oa::U8*>(&header), sizeof(header));
	hashUpdate(hash, sections,
			   static_cast<oa::Usize>(header.numSections) * sizeof(oa::ModelFileSectionHeader));
	return hash;
}

oa::ModelFile makeQuantizableModel() {
	oa::ModelFile model;
	std::strncpy(model.config.architecture, "QuantIntegrityTest",
				 sizeof(model.config.architecture) - 1);
	oa::Vec<oa::F32> weights(32, 0.0F);
	for (oa::I32 value = -7; value <= 7; ++value) {
		weights[static_cast<oa::Usize>(value + 7)] = static_cast<oa::F32>(value);
	}
	const oa::U64 weightShape[] = {2, 16};
	model.addWeight("linear.weight", oa::ScalarType::Float32, {weightShape, 2}, weights.data(),
					weights.size() * sizeof(oa::F32));
	const oa::F32 bias = 0.25F;
	const oa::U64 biasShape[] = {1};
	model.addWeight("linear.bias", oa::ScalarType::Float32, {biasShape, 1}, &bias, sizeof(bias));
	model.optimizerPresent = true;
	model.adamM = {1.0F};
	model.adamV = {2.0F};
	return model;
}

class ModelFileIntegrityTest : public ::testing::Test {
protected:
	void SetUp() override {
		directory = makeDirectory();
		ASSERT_TRUE(oa::Filesystem::createDirectories(directory).isOk());
	}

	void TearDown() override { (void)oa::Filesystem::removeDirectory(directory, true); }

	oa::Path file(const char* inName) const { return directory / oa::Path(inName); }

	oa::Path directory;
};

class EmptyCheckpointModule final : public oa::Module {
public:
	EmptyCheckpointModule() = default;
};

template <typename T>
concept HasAmbientModuleSave =
	requires(const T& inModule, const oa::String& inPath) { inModule.save(inPath); };

template <typename T>
concept HasExplicitModuleSave =
	requires(const T& inModule, oa::Engine& inEngine, const oa::String& inPath) {
		inModule.save(inEngine, inPath);
	};

template <typename T>
concept HasAmbientOptimizerSave =
	requires(const T& inOptimizer, oa::ModelFile& outModel) { inOptimizer.saveTo(outModel); };

template <typename T>
concept HasExplicitOptimizerSave =
	requires(const T& inOptimizer, oa::Engine& inEngine, oa::ModelFile& outModel) {
		inOptimizer.saveTo(inEngine, outModel);
	};

static_assert(not HasAmbientModuleSave<EmptyCheckpointModule>);
static_assert(HasExplicitModuleSave<EmptyCheckpointModule>);
static_assert(not HasAmbientOptimizerSave<oa::OptimizerNoOp>);
static_assert(HasExplicitOptimizerSave<oa::OptimizerNoOp>);
static_assert(not std::is_constructible_v<oa::CheckpointManager, oa::CheckpointManagerConfig>);
static_assert(
	std::is_constructible_v<oa::CheckpointManager, oa::Engine&, oa::CheckpointManagerConfig>);

void expectCheckpointCorrupt(const oa::Result<oa::ModelFile>& inResult) {
	ASSERT_FALSE(inResult.isOk());
	EXPECT_EQ(inResult.getStatus().getCode(), oa::StatusCode::CheckpointCorrupt)
		<< inResult.getStatus().toString().cStr();
}

} // namespace

TEST_F(ModelFileIntegrityTest, V3RoundTripAndAtomicReplacement) {
	const auto path = file("model.oam");
	ASSERT_TRUE(makeModel().save(path.string()).isOk());
	auto first = oa::ModelFile::load(path.string());
	ASSERT_TRUE(first.isOk()) << first.getStatus().toString().cStr();
	ASSERT_EQ(first->weightIndex.size(), 1U);
	const oa::I64 firstStep = first->progress.step;
	EXPECT_EQ(firstStep, 17);

	ASSERT_TRUE(makeModel(9.0F).save(path.string()).isOk());
	auto second = oa::ModelFile::load(path.string());
	ASSERT_TRUE(second.isOk()) << second.getStatus().toString().cStr();
	const auto* values = static_cast<const oa::F32*>(second->weightPtr("linear.weight"));
	ASSERT_NE(values, nullptr);
	EXPECT_FLOAT_EQ(values[3], 9.0F);

	auto bytes = oa::Filesystem::readBinary(path);
	ASSERT_TRUE(bytes.isOk());
	const auto* header = reinterpret_cast<const oa::ModelFileHeader*>(bytes->data());
	const oa::U32 version = header->version;
	EXPECT_EQ(version, oa::kModelFileVersion);
}

TEST_F(ModelFileIntegrityTest, RejectsPlausibleFinitePayloadBitFlip) {
	const auto validPath = file("valid.oam");
	const auto corruptPath = file("payload-corrupt.oam");
	ASSERT_TRUE(makeModel().save(validPath.string()).isOk());
	auto bytes = oa::Filesystem::readBinary(validPath);
	ASSERT_TRUE(bytes.isOk());
	const auto* header = reinterpret_cast<const oa::ModelFileHeader*>(bytes->data());
	const auto* sections = reinterpret_cast<const oa::ModelFileSectionHeader*>(
		bytes->data() + sizeof(oa::ModelFileHeader));
	oa::U64 payloadOffset = 0;
	for (oa::U32 i = 0; i < header->numSections; ++i) {
		if (sections[i].type == static_cast<oa::U32>(oa::ModelFileSection::Weights)) {
			payloadOffset = sections[i].offset + sections[i].size - 1;
		}
	}
	ASSERT_NE(payloadOffset, 0U);
	bytes->at(static_cast<oa::Usize>(payloadOffset)) ^= 0x01U;
	ASSERT_TRUE(oa::Filesystem::writeBinary(corruptPath, {bytes->data(), bytes->size()}).isOk());
	expectCheckpointCorrupt(oa::ModelFile::load(corruptPath.string()));
}

TEST_F(ModelFileIntegrityTest, RejectsMetadataMutationAndTruncation) {
	const auto validPath = file("valid.oam");
	ASSERT_TRUE(makeModel().save(validPath.string()).isOk());
	auto bytes = oa::Filesystem::readBinary(validPath);
	ASSERT_TRUE(bytes.isOk());

	auto metadata = *bytes;
	auto* sections = reinterpret_cast<oa::ModelFileSectionHeader*>(metadata.data() +
																   sizeof(oa::ModelFileHeader));
	sections[0].flags ^= 1U;
	const auto metadataPath = file("metadata-corrupt.oam");
	ASSERT_TRUE(
		oa::Filesystem::writeBinary(metadataPath, {metadata.data(), metadata.size()}).isOk());
	expectCheckpointCorrupt(oa::ModelFile::load(metadataPath.string()));

	bytes->resize(bytes->size() - 7);
	const auto truncatedPath = file("truncated.oam");
	ASSERT_TRUE(oa::Filesystem::writeBinary(truncatedPath, {bytes->data(), bytes->size()}).isOk());
	expectCheckpointCorrupt(oa::ModelFile::load(truncatedPath.string()));
}

TEST_F(ModelFileIntegrityTest, VerifiesAndLoadsLegacyV1Checksum) {
	const auto validPath = file("v2.oam");
	ASSERT_TRUE(makeModel().save(validPath.string()).isOk());
	auto bytes = oa::Filesystem::readBinary(validPath);
	ASSERT_TRUE(bytes.isOk());
	auto* header = reinterpret_cast<oa::ModelFileHeader*>(bytes->data());
	header->version = 1;
	header->checksum = legacyFileChecksum(*bytes);
	const auto legacyPath = file("v1.oam");
	ASSERT_TRUE(oa::Filesystem::writeBinary(legacyPath, {bytes->data(), bytes->size()}).isOk());
	auto loaded = oa::ModelFile::load(legacyPath.string());
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString().cStr();
	EXPECT_NE(loaded->weightPtr("linear.weight"), nullptr);
}

TEST_F(ModelFileIntegrityTest, VerifiesAndLoadsLegacyV2ManifestChecksum) {
	const auto validPath = file("v3.oam");
	ASSERT_TRUE(makeModel().save(validPath.string()).isOk());
	auto bytes = oa::Filesystem::readBinary(validPath);
	ASSERT_TRUE(bytes.isOk());
	auto* header = reinterpret_cast<oa::ModelFileHeader*>(bytes->data());
	header->version = 2;
	header->checksum = manifestChecksum(*bytes);
	const auto legacyPath = file("v2.oam");
	ASSERT_TRUE(oa::Filesystem::writeBinary(legacyPath, {bytes->data(), bytes->size()}).isOk());
	auto loaded = oa::ModelFile::load(legacyPath.string());
	ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString().cStr();
	ASSERT_EQ(loaded->weightIndex.size(), 1U);
	EXPECT_EQ(loaded->weightIndex[0].encoding, oa::ModelTensorEncoding::Dense);
	EXPECT_EQ(loaded->weightIndex[0].blockSize, 0);
}

TEST_F(ModelFileIntegrityTest, QuantizedInferenceArtifactsHaveExactNativeLayouts) {
	for (const auto quantization : {oa::Quantization::Q4, oa::Quantization::Q8}) {
		auto converted = makeQuantizableModel().quantizeWeights(quantization);
		ASSERT_TRUE(converted.isOk()) << converted.getStatus().toString().cStr();
		ASSERT_EQ(converted->weightIndex.size(), 2U);
		const auto* weight = converted->findWeight("linear.weight");
		const auto* bias = converted->findWeight("linear.bias");
		ASSERT_NE(weight, nullptr);
		ASSERT_NE(bias, nullptr);
		const auto encoding = quantization == oa::Quantization::Q4 ? oa::ModelTensorEncoding::Q4
																   : oa::ModelTensorEncoding::Q8;
		EXPECT_EQ(weight->encoding, encoding);
		EXPECT_EQ(weight->blockSize, 32);
		EXPECT_EQ(weight->dtype, oa::ScalarType::Float32);
		EXPECT_EQ(weight->numBytes, quantization == oa::Quantization::Q4 ? 20U : 36U);
		EXPECT_EQ(bias->encoding, oa::ModelTensorEncoding::Dense);
		EXPECT_EQ(bias->numBytes, sizeof(oa::F32));
		EXPECT_FALSE(converted->hasOptimizer());

		const auto* bytes = static_cast<const oa::U8*>(converted->weightPtr("linear.weight"));
		ASSERT_NE(bytes, nullptr);
		const oa::U64 payloadBytes = quantization == oa::Quantization::Q4 ? 16U : 32U;
		oa::F32 scale = 0.0F;
		std::memcpy(&scale, bytes + payloadBytes, sizeof(scale));
		if (quantization == oa::Quantization::Q4) {
			EXPECT_FLOAT_EQ(scale, 1.0F);
			EXPECT_EQ(bytes[0] & 0x0FU, 0U);
			EXPECT_EQ(bytes[0] >> 4U, 1U);
			EXPECT_EQ(bytes[7] & 0x0FU, 14U);
			EXPECT_EQ(bytes[7] >> 4U, 7U);
		} else {
			EXPECT_FLOAT_EQ(scale, 7.0F / 127.0F);
			EXPECT_EQ(bytes[0], 0x81U);
			EXPECT_EQ(bytes[7], 0U);
			EXPECT_EQ(bytes[14], 0x7FU);
			EXPECT_EQ(bytes[15], 0U);
		}

		const auto path =
			file(quantization == oa::Quantization::Q4 ? "model-q4.oam" : "model-q8.oam");
		ASSERT_TRUE(converted->save(path.string()).isOk());
		auto loaded = oa::ModelFile::load(path.string());
		ASSERT_TRUE(loaded.isOk()) << loaded.getStatus().toString().cStr();
		ASSERT_NE(loaded->findWeight("linear.weight"), nullptr);
		EXPECT_EQ(loaded->findWeight("linear.weight")->encoding, encoding);
	}
}

TEST_F(ModelFileIntegrityTest, RejectsMalformedQuantizedTensorMetadata) {
	auto model = makeQuantizableModel();
	auto& weight = model.weightIndex[0];
	weight.encoding = oa::ModelTensorEncoding::Q4;
	weight.blockSize = 32;
	const auto status = model.save(file("malformed.oam").string());
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
}

TEST_F(ModelFileIntegrityTest, RefusesToSaveOverlappingTensorPayloads) {
	auto model = makeModel();
	const oa::F32 bias = 0.25F;
	const oa::U64 shape[] = {1};
	model.addWeight("linear.bias", oa::ScalarType::Float32, {shape, 1}, &bias, sizeof(bias));
	ASSERT_EQ(model.weightIndex.size(), 2U);
	model.weightIndex[1].blobOffset = 0;
	const auto status = model.save(file("overlap.oam").string());
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
}

TEST_F(ModelFileIntegrityTest, VerifiesAdamAndMuonOptimizerPayloads) {
	for (const char* type : {"AdamW", "Muon"}) {
		auto model = makeModel();
		std::memset(model.optimizer.type, 0, sizeof(model.optimizer.type));
		std::strncpy(model.optimizer.type, type, sizeof(model.optimizer.type) - 1);
		model.adamM = {1.0F, 2.0F, 3.0F};
		if (std::strcmp(type, "Muon") != 0)
			model.adamV = {4.0F, 5.0F, 6.0F};

		const auto path = file((oa::String(type) + ".oam").cStr());
		ASSERT_TRUE(model.save(path.string()).isOk());
		auto loaded = oa::ModelFile::load(path.string());
		ASSERT_TRUE(loaded.isOk()) << type << ": " << loaded.getStatus().toString().cStr();
		EXPECT_EQ(loaded->adamM.size(), 3U);
		EXPECT_EQ(loaded->adamV.size(), std::strcmp(type, "Muon") == 0 ? 0U : 3U);
	}
}

TEST_F(ModelFileIntegrityTest, RefusesUnknownOptimizerEncoding) {
	auto model = makeModel();
	model.optimizerPresent = true;
	std::memset(model.optimizer.type, 0, sizeof(model.optimizer.type));
	std::strncpy(model.optimizer.type, "Unknown", sizeof(model.optimizer.type) - 1);
	const auto status = model.save(file("unknown.oam").string());
	EXPECT_FALSE(status.isOk());
	EXPECT_EQ(status.getCode(), oa::StatusCode::InvalidArgument);
}
