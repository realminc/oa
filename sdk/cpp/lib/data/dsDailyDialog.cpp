// oa::DsDailyDialog — DailyDialog conversational text dataset implementation.

#include <data/dsDailyDialog.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/log.h>
#include <fstream>
#include <algorithm>

namespace oa {

DsDailyDialog::DsDailyDialog(const oa::String& inFilePath, oa::I32 inBatchSize,
	oa::I32 inContextLen, bool inShuffle)
	: filePath_(inFilePath)
	, batchSize_(inBatchSize)
	, contextLen_(inContextLen)
	, shuffle_(inShuffle)
	, rng_(42)
{
	if (batchSize_ <= 0 || contextLen_ <= 0) {
		OaLogError(oa::LogComponent::Data, "DailyDialog batch size and context length must be positive");
		return;
	}
	if (!loadDataset()) {
		OaLogError(oa::LogComponent::Data,
			"Failed to load DailyDialog from: %s", inFilePath.cStr());
		return;
	}

	const oa::I64 validStarts = std::max<oa::I64>(0, numChars_ - contextLen_);
	numSamples_ = validStarts;
	numBatches_ = (validStarts + batchSize_ - 1) / batchSize_;

	indices_.resize(validStarts);
	for (oa::I64 i = 0; i < validStarts; ++i) indices_[i] = i;

	if (shuffle_) {
		std::shuffle(indices_.begin(), indices_.end(), rng_);
	}
}

bool DsDailyDialog::loadDataset() {
	std::ifstream file(filePath_.cStr(), std::ios::binary);
	if (!file) {
		OaLogError(oa::LogComponent::Data,
			"Failed to open DailyDialog file: %s", filePath_.cStr());
		return false;
	}

	file.seekg(0, std::ios::end);
	const oa::I64 fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	text_.resize(fileSize);
	file.read(reinterpret_cast<char*>(text_.data()), fileSize);

	if (!file) {
		OaLogError(oa::LogComponent::Data,
			"Failed to read DailyDialog file: %s", filePath_.cStr());
		return false;
	}

	numChars_ = fileSize;
	numConversations_ = 0;

	// Count conversations (blank line separated blocks)
	bool inConversation = false;
	for (oa::I64 i = 0; i < fileSize; ++i) {
		if (text_[i] == '\n') {
			if (i + 1 < fileSize && text_[i + 1] == '\n') {
				if (inConversation) {
					numConversations_++;
					inConversation = false;
				}
			} else {
				inConversation = true;
			}
		} else if (text_[i] != '\r') {
			inConversation = true;
		}
	}
	if (inConversation) {
		numConversations_++;
	}

	return true;
}

oa::Matrix DsDailyDialog::getItem(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= numSamples_) return {};
	oa::Span<const oa::U8> slice(text_.data() + inIndex, contextLen_);
	return oa::FnMatrix::fromBytes(slice, oa::MatrixShape{contextLen_}, oa::ScalarType::UInt8);
}

Dataset::Sample DsDailyDialog::getSample(oa::I64 inIndex) const {
	if (inIndex < 0 || inIndex >= numSamples_) return {};
	oa::Span<const oa::U8> xSlice(text_.data() + inIndex, contextLen_);
	oa::Span<const oa::U8> ySlice(text_.data() + inIndex + contextLen_, 1);
	return Sample(
		oa::FnMatrix::fromBytes(xSlice, oa::MatrixShape{contextLen_}, oa::ScalarType::UInt8),
		oa::FnMatrix::fromBytes(ySlice, oa::MatrixShape{1}, oa::ScalarType::UInt8)
	);
}

bool DsDailyDialog::nextBatch(oa::Matrix& outX, oa::Matrix& outY) {
	if (currentBatch_ >= numBatches_) {
		return false;
	}

	oa::I64 startIdx = static_cast<oa::I64>(currentBatch_) * batchSize_;
	oa::I64 endIdx = std::min(startIdx + batchSize_, static_cast<oa::I64>(indices_.size()));
	oa::I32 actualBatchSize = static_cast<oa::I32>(endIdx - startIdx);

	oa::Vec<oa::U8> xBuf(static_cast<oa::I64>(actualBatchSize) * contextLen_);
	oa::Vec<oa::U8> yBuf(actualBatchSize);

	for (oa::I32 i = 0; i < actualBatchSize; ++i) {
		oa::I64 startPos = indices_[startIdx + i];
		for (oa::I32 t = 0; t < contextLen_; ++t) {
			xBuf[static_cast<oa::I64>(i) * contextLen_ + t] = text_[startPos + t];
		}
		yBuf[i] = text_[startPos + contextLen_];
	}

	outX = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(xBuf.data(), xBuf.size()),
		oa::MatrixShape{actualBatchSize, contextLen_},
		oa::ScalarType::UInt8);
	outY = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(yBuf.data(), yBuf.size()),
		oa::MatrixShape{actualBatchSize},
		oa::ScalarType::UInt8);

	++currentBatch_;
	return true;
}

void DsDailyDialog::reset(bool inReshuffle) {
	currentBatch_ = 0;
	if (inReshuffle && shuffle_) {
		std::shuffle(indices_.begin(), indices_.end(), rng_);
	}
}

} // namespace oa
