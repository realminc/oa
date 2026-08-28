// oa::DsDailyDialog — DailyDialog conversational text dataset (oa::Dataset subclass)
//
// Loads plain-text conversations (speaker labels, blank-line separated blocks).
// Each sample is a context window of N bytes + the next byte as label.
//
// usage:
//   oa::DsDailyDialog ds("/data/dailydialog/train.txt", 64, 128);
//   oa::DataLoader loader(ds, {.batchSize = 64, .shuffle = true});
//   while (auto batch = loader.nextBatch()) { ... }
//
// Legacy convenience batching is also available directly on the class.

#pragma once

#include <oa/data/dataset.h>
#include <oa/core/types.h>
#include <oa/core/matrix.h>
#include <oa/core/status.h>
#include <random>

namespace oa {

class DsDailyDialog : public Dataset {
public:
	/// load DailyDialog dataset from text file.
	/// @param inFilePath Path to train.txt, val.txt, or test.txt
	/// @param inBatchSize Batch size for legacy nextBatch()
	/// @param inContextLen context window length (bytes to predict next)
	/// @param inShuffle Enable shuffling on reset()
	DsDailyDialog(const oa::String& inFilePath, oa::I32 inBatchSize = 64, oa::I32 inContextLen = 128, bool inShuffle = true);

	// Dataset interface
	[[nodiscard]] oa::I64 size() const override { return numSamples_; }
	[[nodiscard]] Matrix getItem(oa::I64 inIndex) const override;
	[[nodiscard]] Sample getSample(oa::I64 inIndex) const override;

	// Convenience: legacy batching interface (keeps old tutorials working).
	bool nextBatch(Matrix& outX, Matrix& outY);
	void reset(bool inReshuffle = true);

	// Metadata
	[[nodiscard]] oa::I64 numChars() const { return numChars_; }
	[[nodiscard]] oa::I32 numConversations() const { return numConversations_; }
	[[nodiscard]] oa::I64 numBatches() const { return numBatches_; }
	[[nodiscard]] oa::I32 contextLen() const { return contextLen_; }

private:
	bool loadDataset();

	oa::String filePath_;
	oa::I32    batchSize_;
	oa::I32    contextLen_;
	bool     shuffle_;
	oa::Vector<oa::U8> text_;    // All text concatenated (UTF-8 bytes)
	oa::I64    numChars_ = 0;
	oa::I32    numConversations_ = 0;
	oa::I64    numSamples_ = 0;
	oa::I64    numBatches_ = 0;

	oa::Vector<oa::I64> indices_;  // Shuffled start positions
	oa::I64        currentBatch_ = 0;
	std::mt19937 rng_;
};

} // namespace oa
