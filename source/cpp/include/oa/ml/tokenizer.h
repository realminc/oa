// OA ML — tokenizer
//
// text tokenization for language models.
// Supports byte-level and BPE (Byte Pair encoding) tokenization.

#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>
#include <oa/core/std/vector.h>

namespace oa {

// ─── BpeTokenizer ────────────────────────────────────────────────────────────
// Learns byte pair merges to build vocabulary from 256 base bytes to target size.
// Useful for compressing sequences and improving model efficiency.

class BpeTokenizer {
public:
	struct Merge {
		oa::U32 left = 0;
		oa::U32 right = 0;
	};

	/// Create BPE tokenizer with target vocabulary size
	explicit BpeTokenizer(oa::I32 inTargetVocab = 512);

	/// Train BPE merges on text corpus
	void train(const char* inText, oa::I32 inNumMerges);

	/// Encode text to BPE tokens
	[[nodiscard]] oa::Vector<oa::I32> encode(const char* inText) const;

	/// Decode BPE tokens back to text
	[[nodiscard]] oa::String decode(const oa::Vector<oa::I32>& inTokens) const;

	/// Persist/load the learned merge ranks. The format is deterministic and
	/// architecture-independent so a training checkpoint can ship its text vocab.
	[[nodiscard]] oa::Status save(const oa::String& inPath) const;
	[[nodiscard]] oa::Status load(const oa::String& inPath);

	/// Encode prompt with padding to context length
	[[nodiscard]] oa::Vector<oa::I32> encodePrompt(const char* inPrompt, oa::I32 inContextLen) const;

	/// get current vocabulary size
	[[nodiscard]] oa::I32 vocabSize() const { return 256 + static_cast<oa::I32>(merges_.size()); }

	/// get number of learned merges
	[[nodiscard]] oa::I32 numMerges() const { return static_cast<oa::I32>(merges_.size()); }

private:
	[[nodiscard]] static oa::Vector<oa::I32> applyMerge(
		const oa::Vector<oa::I32>& inIds, const Merge& inMerge, oa::I32 inNewToken);
	void appendDecoded(oa::I32 inToken, oa::String& outText) const;

	oa::I32 targetVocab_;
	// Merge rank i creates token 256+i. Operands are full token IDs, not bytes:
	// later merges may legally reference earlier learned tokens.
	oa::Vector<Merge> merges_;
};

// Legacy alias — remove once call sites are migrated.

} // namespace oa
