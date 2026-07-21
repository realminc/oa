#include "../../OaTest.h"

#include <Oa/Ml/Oam.h>
#include <Oa/Ml/Checkpoint.h>
#include <Oa/Ml/Module.h>
#include <Oa/Ml/Optim.h>

#include <chrono>
#include <cstring>

namespace {

OaPath MakeDirectory() {
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	return OaFileIo::GetTempDirectory() /
		OaPath("oa_oam_integrity_" + std::to_string(static_cast<long long>(tick)));
}

OamModel MakeModel(OaF32 InLastValue = 4.0F) {
	OamModel model;
	std::strncpy(model.Config.Architecture, "IntegrityTest",
		sizeof(model.Config.Architecture) - 1);
	model.Config.DModel = 4;
	model.Progress.Step = 17;
	const OaF32 weights[] = {1.0F, 2.0F, 3.0F, InLastValue};
	const OaU64 shape[] = {4};
	model.AddWeight("linear.weight", OaScalarType::Float32,
		{shape, 1}, weights, sizeof(weights));
	return model;
}

OaU64 LegacyFileChecksum(const OaVec<OaU8>& InBytes) {
	const auto* header = reinterpret_cast<const OamFileHeader*>(InBytes.Data());
	const auto* sections = reinterpret_cast<const OamSectionHeader*>(
		InBytes.Data() + sizeof(OamFileHeader));
	OaU64 checksum = 0;
	for (OaU32 i = 0; i < header->NumSections; ++i) checksum ^= sections[i].Checksum;
	return checksum;
}

class OamIntegrityTest : public ::testing::Test {
protected:
	void SetUp() override {
		Directory = MakeDirectory();
		ASSERT_TRUE(OaFileIo::CreateDirectories(Directory).IsOk());
	}

	void TearDown() override {
		(void)OaFileIo::RemoveDirectory(Directory, true);
	}

	OaPath File(const char* InName) const { return Directory / OaPath(InName); }

	OaPath Directory;
};

class EmptyCheckpointModule final : public OaModule {
public:
	EmptyCheckpointModule() = default;
};

void ExpectCheckpointCorrupt(const OaResult<OamModel>& InResult) {
	ASSERT_FALSE(InResult.IsOk());
	EXPECT_EQ(InResult.GetStatus().GetCode(), OaStatusCode::CheckpointCorrupt)
		<< InResult.GetStatus().ToString().CStr();
}

} // namespace

TEST_F(OamIntegrityTest, V2RoundTripAndAtomicReplacement) {
	const auto path = File("model.oam");
	ASSERT_TRUE(MakeModel().Save(path.String()).IsOk());
	auto first = OamModel::Load(path.String());
	ASSERT_TRUE(first.IsOk()) << first.GetStatus().ToString().CStr();
	ASSERT_EQ(first->WeightIndex.Size(), 1U);
	const OaI64 firstStep = first->Progress.Step;
	EXPECT_EQ(firstStep, 17);

	ASSERT_TRUE(MakeModel(9.0F).Save(path.String()).IsOk());
	auto second = OamModel::Load(path.String());
	ASSERT_TRUE(second.IsOk()) << second.GetStatus().ToString().CStr();
	const auto* values = static_cast<const OaF32*>(second->WeightPtr("linear.weight"));
	ASSERT_NE(values, nullptr);
	EXPECT_FLOAT_EQ(values[3], 9.0F);

	auto bytes = OaFileIo::ReadBinary(path);
	ASSERT_TRUE(bytes.IsOk());
	const auto* header = reinterpret_cast<const OamFileHeader*>(bytes->Data());
	const OaU32 version = header->Version;
	EXPECT_EQ(version, OAM_VERSION);
}

TEST_F(OamIntegrityTest, RejectsPlausibleFinitePayloadBitFlip) {
	const auto validPath = File("valid.oam");
	const auto corruptPath = File("payload-corrupt.oam");
	ASSERT_TRUE(MakeModel().Save(validPath.String()).IsOk());
	auto bytes = OaFileIo::ReadBinary(validPath);
	ASSERT_TRUE(bytes.IsOk());
	const auto* header = reinterpret_cast<const OamFileHeader*>(bytes->Data());
	const auto* sections = reinterpret_cast<const OamSectionHeader*>(
		bytes->Data() + sizeof(OamFileHeader));
	OaU64 payloadOffset = 0;
	for (OaU32 i = 0; i < header->NumSections; ++i) {
		if (sections[i].Type == static_cast<OaU32>(OamSectionType::Weights)) {
			payloadOffset = sections[i].Offset + sections[i].Size - 1;
		}
	}
	ASSERT_NE(payloadOffset, 0U);
	bytes->At(static_cast<OaUsize>(payloadOffset)) ^= 0x01U;
	ASSERT_TRUE(OaFileIo::WriteBinary(corruptPath,
		{bytes->Data(), bytes->Size()}).IsOk());
	ExpectCheckpointCorrupt(OamModel::Load(corruptPath.String()));
}

TEST_F(OamIntegrityTest, RejectsMetadataMutationAndTruncation) {
	const auto validPath = File("valid.oam");
	ASSERT_TRUE(MakeModel().Save(validPath.String()).IsOk());
	auto bytes = OaFileIo::ReadBinary(validPath);
	ASSERT_TRUE(bytes.IsOk());

	auto metadata = *bytes;
	auto* sections = reinterpret_cast<OamSectionHeader*>(
		metadata.Data() + sizeof(OamFileHeader));
	sections[0].Flags ^= 1U;
	const auto metadataPath = File("metadata-corrupt.oam");
	ASSERT_TRUE(OaFileIo::WriteBinary(metadataPath,
		{metadata.Data(), metadata.Size()}).IsOk());
	ExpectCheckpointCorrupt(OamModel::Load(metadataPath.String()));

	bytes->Resize(bytes->Size() - 7);
	const auto truncatedPath = File("truncated.oam");
	ASSERT_TRUE(OaFileIo::WriteBinary(truncatedPath,
		{bytes->Data(), bytes->Size()}).IsOk());
	ExpectCheckpointCorrupt(OamModel::Load(truncatedPath.String()));
}

TEST_F(OamIntegrityTest, VerifiesAndLoadsLegacyV1Checksum) {
	const auto validPath = File("v2.oam");
	ASSERT_TRUE(MakeModel().Save(validPath.String()).IsOk());
	auto bytes = OaFileIo::ReadBinary(validPath);
	ASSERT_TRUE(bytes.IsOk());
	auto* header = reinterpret_cast<OamFileHeader*>(bytes->Data());
	header->Version = 1;
	header->Checksum = LegacyFileChecksum(*bytes);
	const auto legacyPath = File("v1.oam");
	ASSERT_TRUE(OaFileIo::WriteBinary(legacyPath,
		{bytes->Data(), bytes->Size()}).IsOk());
	auto loaded = OamModel::Load(legacyPath.String());
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().ToString().CStr();
	EXPECT_NE(loaded->WeightPtr("linear.weight"), nullptr);
}

TEST_F(OamIntegrityTest, VerifiesAdamMuonAndCompositeOptimizerPayloads) {
	for (const char* type : {"AdamW", "Muon", "MuonAdamW"}) {
		auto model = MakeModel();
		std::memset(model.Optimizer.Type, 0, sizeof(model.Optimizer.Type));
		std::strncpy(model.Optimizer.Type, type, sizeof(model.Optimizer.Type) - 1);
		model.AdamM = {1.0F, 2.0F, 3.0F};
		if (std::strcmp(type, "Muon") != 0) model.AdamV = {4.0F, 5.0F, 6.0F};
		if (std::strcmp(type, "MuonAdamW") == 0) model.MuonM = {7.0F, 8.0F};

		const auto path = File((OaString(type) + ".oam").CStr());
		ASSERT_TRUE(model.Save(path.String()).IsOk());
		auto loaded = OamModel::Load(path.String());
		ASSERT_TRUE(loaded.IsOk()) << type << ": "
			<< loaded.GetStatus().ToString().CStr();
		EXPECT_EQ(loaded->AdamM.Size(), 3U);
		EXPECT_EQ(loaded->AdamV.Size(),
			std::strcmp(type, "Muon") == 0 ? 0U : 3U);
		EXPECT_EQ(loaded->MuonM.Size(),
			std::strcmp(type, "MuonAdamW") == 0 ? 2U : 0U);
	}
}

TEST_F(OamIntegrityTest, PersistsHeaderOnlySgdOptimizerState) {
	OaVec<OaParameter> parameters;
	OaSGD source(parameters, 0.125F, 0.75F, 0.01F);
	OamModel model = MakeModel();
	ASSERT_TRUE(source.SaveTo(model).IsOk());
	const auto path = File("sgd.oam");
	ASSERT_TRUE(model.Save(path.String()).IsOk());
	auto loaded = OamModel::Load(path.String());
	ASSERT_TRUE(loaded.IsOk()) << loaded.GetStatus().ToString().CStr();
	EXPECT_TRUE(loaded->HasOptimizer());
	EXPECT_STREQ(loaded->Optimizer.Type, "SGD");
	OaSGD restored(parameters);
	ASSERT_TRUE(restored.LoadFrom(*loaded).IsOk());
	EXPECT_FLOAT_EQ(restored.GetLr(), 0.125F);
}

TEST_F(OamIntegrityTest, RefusesUnknownOptimizerEncoding) {
	auto model = MakeModel();
	model.OptimizerPresent = true;
	std::memset(model.Optimizer.Type, 0, sizeof(model.Optimizer.Type));
	std::strncpy(model.Optimizer.Type, "Unknown",
		sizeof(model.Optimizer.Type) - 1);
	const auto status = model.Save(File("unknown.oam").String());
	EXPECT_FALSE(status.IsOk());
	EXPECT_EQ(status.GetCode(), OaStatusCode::InvalidArgument);
}

TEST_F(OamIntegrityTest, CheckpointManagerPersistsAndVerifiesStepLineage) {
	EmptyCheckpointModule module;
	OaOptimizerNoOp optimizer;
	OaCheckpointManager manager({
		.Dir = Directory.String(),
		.ModelName = "Lineage",
		.MaxKeep = 2,
	});
	ASSERT_TRUE(manager.SaveIncremental(module, optimizer, 42, 0.5).IsOk());
	auto files = OaFileIo::ListFiles(
		OaPath(manager.GetIncrementalDir()), ".oam");
	ASSERT_TRUE(files.IsOk());
	ASSERT_EQ(files->Size(), 1U);
	auto loaded = OamModel::Load(files->At(0).String());
	ASSERT_TRUE(loaded.IsOk());
	const OaI64 savedStep = loaded->Progress.Step;
	EXPECT_EQ(savedStep, 42);
	EXPECT_FLOAT_EQ(loaded->Progress.BestMetric, 0.5F);

	// This is a structurally valid v2 checkpoint whose internal progress does
	// not match the step used to select it. Restore must fail closed rather than
	// silently accepting the wrong recovery boundary.
	loaded->Progress.Step = 41;
	ASSERT_TRUE(loaded->Save(files->At(0).String()).IsOk());
	const auto status = manager.LoadLatestInto(module, optimizer);
	EXPECT_EQ(status.GetCode(), OaStatusCode::CheckpointCorrupt)
		<< status.ToString().CStr();
}
