// OaDsDailyDialog — DailyDialog conversational text dataset (OaDataset subclass)
//
// Loads plain-text conversations (speaker labels, blank-line separated blocks).
// Each sample is a context window of N bytes + the next byte as label.
//
// Usage:
//   OaDsDailyDialog ds("/data/dailydialog/train.txt", 64, 128);
//   OaDataLoader loader(ds, {.BatchSize = 64, .Shuffle = true});
//   while (auto batch = loader.NextBatch()) { ... }
//
// Legacy convenience batching is also available directly on the class.

#pragma once

#include <Oa/Data/Dataset.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Core/Status.h>
#include <random>

class OaDsDailyDialog : public OaDataset {
public:
	/// Load DailyDialog dataset from text file.
	/// @param InFilePath Path to train.txt, val.txt, or test.txt
	/// @param InBatchSize Batch size for legacy NextBatch()
	/// @param InContextLen Context window length (bytes to predict next)
	/// @param InShuffle Enable shuffling on Reset()
	OaDsDailyDialog(const OaString& InFilePath, OaI32 InBatchSize = 64, OaI32 InContextLen = 128, bool InShuffle = true);

	// OaDataset interface
	[[nodiscard]] OaI64 Size() const override { return NumSamples_; }
	[[nodiscard]] OaMatrix GetItem(OaI64 InIndex) const override;
	[[nodiscard]] Sample GetSample(OaI64 InIndex) const override;

	// Convenience: legacy batching interface (keeps old tutorials working).
	bool NextBatch(OaMatrix& OutX, OaMatrix& OutY);
	void Reset(bool InReshuffle = true);

	// Metadata
	[[nodiscard]] OaI64 NumChars() const { return NumChars_; }
	[[nodiscard]] OaI32 NumConversations() const { return NumConversations_; }
	[[nodiscard]] OaI64 NumBatches() const { return NumBatches_; }
	[[nodiscard]] OaI32 ContextLen() const { return ContextLen_; }

private:
	bool LoadDataset();

	OaString FilePath_;
	OaI32    BatchSize_;
	OaI32    ContextLen_;
	bool     Shuffle_;
	OaVec<OaU8> Text_;    // All text concatenated (UTF-8 bytes)
	OaI64    NumChars_ = 0;
	OaI32    NumConversations_ = 0;
	OaI64    NumSamples_ = 0;
	OaI64    NumBatches_ = 0;

	OaVec<OaI64> Indices_;  // Shuffled start positions
	OaI64        CurrentBatch_ = 0;
	std::mt19937 Rng_;
};
