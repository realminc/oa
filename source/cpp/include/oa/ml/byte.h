// OA ML - Byte-level encoding
//
// THE DIFFERENTIATOR. No tokenizer. No vocabulary. No sentencepiece.
// Just bytes. Universal. Fast.
//
// Vocabulary = 256 (one per byte value). Always. Every modality.
//
// text?  Bytes.
// Image? Bytes.
// Audio? Bytes.
// Video? Bytes.
// Code?  Bytes.
// DNA?   Bytes.
//
// Benefits:
//   - Zero preprocessing latency (no tokenization step)
//   - Universal: same model handles any modality
//   - Lossless: no information lost to vocabulary mapping
//   - Simple: embedding table is 256 x d_model (tiny)
//   - Robust: handles any language, encoding, binary format
//
// Inspired by: megaByte (meta), byT5 (Google), but simpler.

#pragma once

#include <oa/ml/module.h>

namespace oa {

// CONSTANTS

/// The one true vocabulary size. 256. Forever.
constexpr I32 ByteVocabSize = 256;

/// Special byte tokens (optional, for sequence control)
constexpr U8 BytePad = 0x00;
constexpr U8 ByteBos = 0x01;
constexpr U8 ByteEos = 0x02;
constexpr U8 ByteSep = 0x03;

// BYTE ENCODER - Raw bytes to tensor and back

class ByteEncoder {
public:
	/// Encode raw bytes to tensor [seq_len] of UInt8
	[[nodiscard]] static oa::Matrix encode(oa::Span<const oa::U8> inBytes);

	/// Encode with batch dimension [1, seq_len]
	[[nodiscard]] static oa::Matrix encodeBatched(oa::Span<const oa::U8> inBytes);

	/// Decode logits [seq_len, 256] back to bytes (argmax)
	[[nodiscard]] static oa::Vector<oa::U8> decode(const oa::Matrix& inLogits);

	/// Decode with temperature sampling
	[[nodiscard]] static Vector<U8> sample(
		const Matrix& inLogits,
		F32 inTemperature = 1.0F,
		F32 inTopP = 0.9F);

	// Multi-Modal Convenience

	/// text: just cast string bytes directly
	[[nodiscard]] static oa::Matrix encodeText(oa::StringView inText) {
		return encode({reinterpret_cast<const oa::U8*>(inText.data()), inText.size()});
	}

	/// Image: raw pixel bytes [H * W * C]
	[[nodiscard]] static oa::Matrix encodeImage(oa::Span<const oa::U8> inPixels, oa::I32 inWidth, oa::I32 inHeight, oa::I32 inChannels);

	/// Audio: raw sample bytes
	[[nodiscard]] static oa::Matrix encodeAudio(oa::Span<const oa::U8> inSamples, oa::I32 inSampleRate, oa::I32 inChannels);

	/// Decode bytes back to string
	[[nodiscard]] static oa::String decodeText(const oa::Matrix& inLogits) {
		auto bytes = decode(inLogits);
		return oa::String(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	}
};

// BYTE EMBEDDING - The 256-entry lookup table
// Maps each byte value (0-255) to a d_model dimensional vector.
// input:  [batch, seq_len] of UInt8
// output: [batch, seq_len, d_model] of Float32
//
// This replaces the massive 30k-100k token embedding tables in GPT/LLaMA.
// Ours is always 256 x d_model. Tiny. Fast. Universal.

class ByteEmbedding : public Module {
public:
	explicit ByteEmbedding(oa::I32 inDModel) : dModel_(inDModel) {
		registerParameter("weight", FnMatrix::randN(
			MatrixShape{ByteVocabSize, inDModel}, FnMatrix::weightDtype()));
	}

	/// [batch, seq] -> [batch, seq, d_model]
	Matrix forward(const Matrix& inByteIds) override;

	[[nodiscard]] oa::I32 dModel() const { return dModel_; }

private:
	oa::I32 dModel_;
};

// BYTE OUTPUT HEAD - convert hidden states back to byte probabilities
// input:  [batch, seq_len, d_model]
// output: [batch, seq_len, 256] (logits over byte values)

class ByteHead : public Module {
public:
	explicit ByteHead(oa::I32 inDModel) : dModel_(inDModel) {
		auto wd = FnMatrix::weightDtype();
		registerParameter("weight", FnMatrix::rand(MatrixShape{ByteVocabSize, inDModel}, wd));
		registerParameter("bias", FnMatrix::zeros(MatrixShape{ByteVocabSize}, wd));
	}

	/// [batch, seq, d_model] -> [batch, seq, 256]
	Matrix forward(const Matrix& inHidden) override;

private:
	oa::I32 dModel_;
};

} // namespace oa
