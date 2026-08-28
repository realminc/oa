#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

struct ClipTokenBatch {
	oa::Vector<oa::I32> tokenIds;
	oa::Vector<oa::I32> flatEosRows;
	oa::I32 batch = 0;
	oa::I32 contextLength = 0;
};

// Native OpenAI CLIP byte-BPE tokenizer. loadMerges consumes the canonical
// bpe_simple_vocab_16e6 merges.txt representation.
class ClipTokenizer {
public:
	ClipTokenizer();
	~ClipTokenizer();
	ClipTokenizer(ClipTokenizer&&) noexcept;
	ClipTokenizer& operator=(ClipTokenizer&&) noexcept;
	ClipTokenizer(const ClipTokenizer&) = delete;
	ClipTokenizer& operator=(const ClipTokenizer&) = delete;

	[[nodiscard]] oa::Status loadMerges(const oa::Path& inPath);
	[[nodiscard]] oa::Status loadMerges(oa::Span<const oa::U8> inBytes);
	[[nodiscard]] oa::Result<ClipTokenBatch> encode(
		oa::Span<const oa::String> inPrompts, oa::I32 inContextLength = 77,
		bool inTruncate = true) const;
	[[nodiscard]] bool isLoaded() const noexcept;
	[[nodiscard]] oa::I32 vocabSize() const noexcept;
	[[nodiscard]] oa::I32 bosToken() const noexcept;
	[[nodiscard]] oa::I32 eosToken() const noexcept;

private:
	class Impl;
	oa::UniquePtr<Impl> impl_;
};

} // namespace oa
