// OA ML - Tokenizer
//
// Text tokenization for language models.
// Supports byte-level and BPE (Byte Pair Encoding) tokenization.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Std/Vec.h>

// ─── Byte Pair Encoding (BPE) Tokenizer ─────────────────────────────────────
// Learns byte pair merges to build vocabulary from 256 base bytes to target size.
// Useful for compressing sequences and improving model efficiency.

class OaBpeTokenizer {
public:
	struct Merge {
		OaU32 Left = 0;
		OaU32 Right = 0;
	};

	/// Create BPE tokenizer with target vocabulary size
	explicit OaBpeTokenizer(OaI32 InTargetVocab = 512);

	/// Train BPE merges on text corpus
	void Train(const char* InText, OaI32 InNumMerges);

	/// Encode text to BPE tokens
	[[nodiscard]] OaVec<OaI32> Encode(const char* InText) const;

	/// Decode BPE tokens back to text
	[[nodiscard]] OaString Decode(const OaVec<OaI32>& InTokens) const;

	/// Persist/load the learned merge ranks. The format is deterministic and
	/// architecture-independent so a training checkpoint can ship its text vocab.
	[[nodiscard]] OaStatus Save(const OaString& InPath) const;
	[[nodiscard]] OaStatus Load(const OaString& InPath);

	/// Encode prompt with padding to context length
	[[nodiscard]] OaVec<OaI32> EncodePrompt(const char* InPrompt, OaI32 InContextLen) const;

	/// Get current vocabulary size
	[[nodiscard]] OaI32 VocabSize() const { return 256 + static_cast<OaI32>(Merges_.Size()); }

	/// Get number of learned merges
	[[nodiscard]] OaI32 NumMerges() const { return static_cast<OaI32>(Merges_.Size()); }

private:
	[[nodiscard]] static OaVec<OaI32> ApplyMerge(
		const OaVec<OaI32>& InIds, const Merge& InMerge, OaI32 InNewToken);
	void AppendDecoded(OaI32 InToken, OaString& OutText) const;

	OaI32 TargetVocab_;
	// Merge rank i creates token 256+i. Operands are full token IDs, not bytes:
	// later merges may legally reference earlier learned tokens.
	OaVec<Merge> Merges_;
};
