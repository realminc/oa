#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

struct OaClipTokenBatch {
	OaVec<OaI32> TokenIds;
	OaVec<OaI32> FlatEosRows;
	OaI32 Batch = 0;
	OaI32 ContextLength = 0;
};

// Native OpenAI CLIP byte-BPE tokenizer. LoadMerges consumes the canonical
// bpe_simple_vocab_16e6 merges.txt representation.
class OaClipTokenizer {
public:
	OaClipTokenizer();
	~OaClipTokenizer();
	OaClipTokenizer(OaClipTokenizer&&) noexcept;
	OaClipTokenizer& operator=(OaClipTokenizer&&) noexcept;
	OaClipTokenizer(const OaClipTokenizer&) = delete;
	OaClipTokenizer& operator=(const OaClipTokenizer&) = delete;

	[[nodiscard]] OaStatus LoadMerges(const OaPath& InPath);
	[[nodiscard]] OaStatus LoadMerges(OaSpan<const OaU8> InBytes);
	[[nodiscard]] OaResult<OaClipTokenBatch> Encode(
		OaSpan<const OaString> InPrompts, OaI32 InContextLength = 77,
		bool InTruncate = true) const;
	[[nodiscard]] bool IsLoaded() const noexcept;
	[[nodiscard]] OaI32 VocabSize() const noexcept;
	[[nodiscard]] OaI32 BosToken() const noexcept;
	[[nodiscard]] OaI32 EosToken() const noexcept;

private:
	class Impl;
	OaUniquePtr<Impl> Impl_;
};
