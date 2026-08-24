#include "../../oaTest.h"

#include <oa/ml/checkpoint.h>
#include <oa/ml/module.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/optim.h>

#include <chrono>

namespace {

class EmptyCheckpointModule final : public oa::Module {};

oa::Path makeCheckpointDirectory() {
	const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
	return oa::Paths::temp() /
		oa::Path("oa_checkpoint_" + std::to_string(static_cast<long long>(tick)));
}

class CheckpointTest : public ::testing::Test {
protected:
	void SetUp() override {
		directory = makeCheckpointDirectory();
		ASSERT_TRUE(oa::Filesystem::createDirectories(directory).isOk());
	}

	void TearDown() override {
		(void)oa::Filesystem::removeDirectory(directory, true);
	}

	oa::Path directory;
};

} // namespace

TEST_VK(CheckpointTest, PersistsHeaderOnlySgdOptimizerState) {
	oa::Vec<oa::Parameter> parameters;
	oa::Sgd source(parameters, 0.125F, 0.75F, 0.01F);
	oa::ModelFile model;
	ASSERT_TRUE(source.saveTo(testEngine(), model).isOk());
	ASSERT_TRUE(model.hasOptimizer());
	EXPECT_STREQ(model.optimizer.type, "SGD");

	oa::Sgd restored(parameters);
	ASSERT_TRUE(restored.loadFrom(testEngine(), model).isOk());
	EXPECT_FLOAT_EQ(restored.getLr(), 0.125F);
}

TEST_VK(CheckpointTest, ManagerPersistsAndVerifiesStepLineage) {
	EmptyCheckpointModule module;
	oa::OptimizerNoOp optimizer;
	oa::CheckpointManager manager(testEngine(), {
		.dir = directory.string(),
		.modelName = "Lineage",
		.context = {},
		.maxKeep = 2,
	});
	ASSERT_TRUE(manager.saveIncremental(module, optimizer, 42, 0.5).isOk());
	auto files = oa::Filesystem::listFiles(
		oa::Path(manager.incrementalDir()), ".oam");
	ASSERT_TRUE(files.isOk());
	ASSERT_EQ(files->size(), 1U);
	auto loaded = oa::ModelFile::load(files->at(0).string());
	ASSERT_TRUE(loaded.isOk());
	const oa::I64 savedStep = loaded->progress.step;
	EXPECT_EQ(savedStep, 42);
	EXPECT_FLOAT_EQ(loaded->progress.bestMetric, 0.5F);

	// A valid v2 file with a mismatched internal progress step must fail closed.
	loaded->progress.step = 41;
	ASSERT_TRUE(loaded->save(files->at(0).string()).isOk());
	const auto status = manager.loadLatestInto(module, optimizer);
	EXPECT_EQ(status.getCode(), oa::StatusCode::CheckpointCorrupt)
		<< status.toString().cStr();
}
