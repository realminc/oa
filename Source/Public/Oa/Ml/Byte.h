// OA ML - Byte-Level Encoding
//
// THE DIFFERENTIATOR. No tokenizer. No vocabulary. No sentencepiece.
// Just bytes. Universal. Fast.
//
// Vocabulary = 256 (one per byte value). Always. Every modality.
//
// Text?  Bytes.
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
// Inspired by: MegaByte (Meta), ByT5 (Google), but simpler.

#pragma once

#include <Oa/Ml/Module.h>

// CONSTANTS

/// The one true vocabulary size. 256. Forever.
constexpr OaI32 OA_BYTE_VOCAB_SIZE = 256;

/// Special byte tokens (optional, for sequence control)
constexpr OaU8 OA_BYTE_PAD = 0x00;   // Padding
constexpr OaU8 OA_BYTE_BOS = 0x01;   // Beginning of sequence
constexpr OaU8 OA_BYTE_EOS = 0x02;   // End of sequence
constexpr OaU8 OA_BYTE_SEP = 0x03;   // Separator (between modalities)

// BYTE ENCODER - Raw bytes to tensor and back

class OaByteEncoder {
public:
	/// Encode raw bytes to tensor [seq_len] of UInt8
	[[nodiscard]] static OaMatrix Encode(OaSpan<const OaU8> InBytes);

	/// Encode with batch dimension [1, seq_len]
	[[nodiscard]] static OaMatrix EncodeBatched(OaSpan<const OaU8> InBytes);

	/// Decode logits [seq_len, 256] back to bytes (argmax)
	[[nodiscard]] static OaVec<OaU8> Decode(const OaMatrix& InLogits);

	/// Decode with temperature sampling
	[[nodiscard]] static OaVec<OaU8> Sample(const OaMatrix& InLogits, OaF32 InTemperature = 1.0f, OaF32 InTopP = 0.9f);

	// Multi-Modal Convenience

	/// Text: just cast string bytes directly
	[[nodiscard]] static OaMatrix EncodeText(OaStringView InText) {
		return Encode({reinterpret_cast<const OaU8*>(InText.data()), InText.size()});
	}

	/// Image: raw pixel bytes [H * W * C]
	[[nodiscard]] static OaMatrix EncodeImage(OaSpan<const OaU8> InPixels, OaI32 InWidth, OaI32 InHeight, OaI32 InChannels);

	/// Audio: raw sample bytes
	[[nodiscard]] static OaMatrix EncodeAudio(OaSpan<const OaU8> InSamples, OaI32 InSampleRate, OaI32 InChannels);

	/// Decode bytes back to string
	[[nodiscard]] static OaString DecodeText(const OaMatrix& InLogits) {
		auto bytes = Decode(InLogits);
		return OaString(reinterpret_cast<const char*>(bytes.Data()), bytes.Size());
	}
};

// BYTE EMBEDDING - The 256-entry lookup table
// Maps each byte value (0-255) to a d_model dimensional vector.
// Input:  [batch, seq_len] of UInt8
// Output: [batch, seq_len, d_model] of Float32
//
// This replaces the massive 30k-100k token embedding tables in GPT/LLaMA.
// Ours is always 256 x d_model. Tiny. Fast. Universal.

class OaByteEmbedding : public OaModule {
public:
	explicit OaByteEmbedding(OaI32 InDModel) : DModel_(InDModel) {
		RegisterParameter("weight", OaFnMatrix::RandN(OaMatrixShape{OA_BYTE_VOCAB_SIZE, InDModel}, OaFnMatrix::GetWeightDtype()));
	}

	/// [batch, seq] -> [batch, seq, d_model]
	OaMatrix Forward(const OaMatrix& InByteIds) override;

	[[nodiscard]] OaI32 DModel() const { return DModel_; }

private:
	OaI32 DModel_;
};

// BYTE OUTPUT HEAD - Convert hidden states back to byte probabilities
// Input:  [batch, seq_len, d_model]
// Output: [batch, seq_len, 256] (logits over byte values)

class OaByteHead : public OaModule {
public:
	explicit OaByteHead(OaI32 InDModel) : DModel_(InDModel) {
		auto wd = OaFnMatrix::GetWeightDtype();
		RegisterParameter("weight", OaFnMatrix::Rand(OaMatrixShape{OA_BYTE_VOCAB_SIZE, InDModel}, wd));
		RegisterParameter("bias", OaFnMatrix::Zeros(OaMatrixShape{OA_BYTE_VOCAB_SIZE}, wd));
	}

	/// [batch, seq, d_model] -> [batch, seq, 256]
	OaMatrix Forward(const OaMatrix& InHidden) override;

private:
	OaI32 DModel_;
};

