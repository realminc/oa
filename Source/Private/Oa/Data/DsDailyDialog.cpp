// OaDsDailyDialog — DailyDialog conversational text dataset implementation

#include <Oa/Data/DsDailyDialog.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Log.h>
#include <fstream>
#include <algorithm>

// ============================================================================
// OaDsDailyDialog Implementation
// ============================================================================

OaDsDailyDialog::OaDsDailyDialog(const OaString& InFilePath, OaI32 InBatchSize,
	OaI32 InContextLen, bool InShuffle)
	: FilePath_(InFilePath)
	, BatchSize_(InBatchSize)
	, ContextLen_(InContextLen)
	, Shuffle_(InShuffle)
	, Rng_(42)
{
	if (BatchSize_ <= 0 || ContextLen_ <= 0) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"DailyDialog batch size and context length must be positive");
		return;
	}
	if (!LoadDataset()) {
		OA_LOG_ERROR(OaLogComponent::ML, "Failed to load DailyDialog from: %s", InFilePath.c_str());
		return;
	}

	const OaI64 validStarts = std::max<OaI64>(0, NumChars_ - ContextLen_);
	NumSamples_ = validStarts;
	NumBatches_ = (validStarts + BatchSize_ - 1) / BatchSize_;

	Indices_.Resize(validStarts);
	for (OaI64 i = 0; i < validStarts; ++i) Indices_[i] = i;

	if (Shuffle_) {
		std::shuffle(Indices_.Begin(), Indices_.End(), Rng_);
	}
}

bool OaDsDailyDialog::LoadDataset() {
	std::ifstream file(FilePath_.c_str(), std::ios::binary);
	if (!file) {
		OA_LOG_ERROR(OaLogComponent::ML, "Failed to open DailyDialog file: %s", FilePath_.c_str());
		return false;
	}

	file.seekg(0, std::ios::end);
	const OaI64 fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	Text_.Resize(fileSize);
	file.read(reinterpret_cast<char*>(Text_.Data()), fileSize);

	if (!file) {
		OA_LOG_ERROR(OaLogComponent::ML, "Failed to read DailyDialog file: %s", FilePath_.c_str());
		return false;
	}

	NumChars_ = fileSize;
	NumConversations_ = 0;

	// Count conversations (blank line separated blocks)
	bool inConversation = false;
	for (OaI64 i = 0; i < fileSize; ++i) {
		if (Text_[i] == '\n') {
			if (i + 1 < fileSize && Text_[i + 1] == '\n') {
				if (inConversation) {
					NumConversations_++;
					inConversation = false;
				}
			} else {
				inConversation = true;
			}
		} else if (Text_[i] != '\r') {
			inConversation = true;
		}
	}
	if (inConversation) {
		NumConversations_++;
	}

	return true;
}

OaMatrix OaDsDailyDialog::GetItem(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= NumSamples_) return {};
	OaSpan<const OaU8> slice(Text_.Data() + InIndex, ContextLen_);
	return OaFnMatrix::FromBytes(slice, OaMatrixShape{ContextLen_}, OaScalarType::UInt8);
}

OaDataset::Sample OaDsDailyDialog::GetSample(OaI64 InIndex) const {
	if (InIndex < 0 || InIndex >= NumSamples_) return {};
	OaSpan<const OaU8> xSlice(Text_.Data() + InIndex, ContextLen_);
	OaSpan<const OaU8> ySlice(Text_.Data() + InIndex + ContextLen_, 1);
	return Sample(
		OaFnMatrix::FromBytes(xSlice, OaMatrixShape{ContextLen_}, OaScalarType::UInt8),
		OaFnMatrix::FromBytes(ySlice, OaMatrixShape{1}, OaScalarType::UInt8)
	);
}

bool OaDsDailyDialog::NextBatch(OaMatrix& OutX, OaMatrix& OutY) {
	if (CurrentBatch_ >= NumBatches_) {
		return false;
	}

	OaI64 startIdx = static_cast<OaI64>(CurrentBatch_) * BatchSize_;
	OaI64 endIdx = std::min(startIdx + BatchSize_, static_cast<OaI64>(Indices_.Size()));
	OaI32 actualBatchSize = static_cast<OaI32>(endIdx - startIdx);

	OaVec<OaU8> xBuf(static_cast<OaI64>(actualBatchSize) * ContextLen_);
	OaVec<OaU8> yBuf(actualBatchSize);

	for (OaI32 i = 0; i < actualBatchSize; ++i) {
		OaI64 startPos = Indices_[startIdx + i];
		for (OaI32 t = 0; t < ContextLen_; ++t) {
			xBuf[static_cast<OaI64>(i) * ContextLen_ + t] = Text_[startPos + t];
		}
		yBuf[i] = Text_[startPos + ContextLen_];
	}

	OutX = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(xBuf.Data(), xBuf.Size()),
		OaMatrixShape{actualBatchSize, ContextLen_},
		OaScalarType::UInt8);
	OutY = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(yBuf.Data(), yBuf.Size()),
		OaMatrixShape{actualBatchSize},
		OaScalarType::UInt8);

	++CurrentBatch_;
	return true;
}

void OaDsDailyDialog::Reset(bool InReshuffle) {
	CurrentBatch_ = 0;
	if (InReshuffle && Shuffle_) {
		std::shuffle(Indices_.Begin(), Indices_.End(), Rng_);
	}
}
