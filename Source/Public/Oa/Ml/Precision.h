// Precision policy for mixed-precision training
//
// Classifies compute shaders by required precision:
//   Bf16  — safe for BF16 storage (default for most forward/backward ops)
//   Fp32  — must use FP32 storage (loss, reductions with large accumulation)
//   Mixed — uses explicit typed helpers, DTYPE-independent
//
// LoadShaders consults this table to set the DTYPE specialization constant:
//   Engine BF16 mode: Bf16 → DTYPE=1 (OaLoad/OaStore use 2-byte elements), Fp32/Mixed → DTYPE=0
//   Engine FP32 mode: all → DTYPE=0
//
// Most ML shaders default to Bf16 so Storage.slang matches OaMatrix BF16 buffers.
// Fp32 entries are for shaders that must read/write FP32 layout only (Mixed helpers, legacy).

#pragma once

#include <Oa/Core/Types.h>

enum class OaPrecisionTag : OaU8 {
	Bf16,
	Fp32,
	Mixed,
};

class OaPrecisionEntry {
public:
	const char* Name;
	OaPrecisionTag Tag;
};

// Shaders NOT listed here default to OaPrecisionTag::Bf16.
// Only list exceptions that require FP32 or mixed-precision handling.
static constexpr OaPrecisionEntry kPrecisionTable[] = {
	// Mixed-precision optimizer — explicit OaLoadF32-style helpers, DTYPE-independent
	{"AdamwMixed", OaPrecisionTag::Mixed},

	// Cast shader — explicit typed helpers
	{"CastF32Bf16", OaPrecisionTag::Mixed},

	// Softmax family — OaLoad/OaStore on logits, must match buffer dtype.
	// Default to Bf16 (same as RmsNorm). Arithmetic is FP32 inside the shader.

	// Reductions — OaLoad/OaStore on inputs, must match buffer dtype.
	// Default to Bf16. Arithmetic is FP32 inside the shader (groupshared float).

	// LayerNorm / RmsNorm — OaLoad/OaStore on activations + weights.
	// Default to Bf16 (AllocF = BF16 when engine BF16).
};

static constexpr OaU32 kPrecisionTableSize =
	sizeof(kPrecisionTable) / sizeof(kPrecisionTable[0]);

[[nodiscard]] static inline OaPrecisionTag OaLookupPrecision(OaStringView InName) {
	for (OaU32 i = 0; i < kPrecisionTableSize; ++i) {
		if (InName.Equals(OaStdStringView(kPrecisionTable[i].Name))) return kPrecisionTable[i].Tag;
	}
	return OaPrecisionTag::Bf16;
}

[[nodiscard]] static inline OaU32 OaPrecisionDtype(
	OaPrecisionTag InTag, OaPrecision InEnginePrecision)
{
	if (InEnginePrecision != OaPrecision::BF16) return 0;
	return (InTag == OaPrecisionTag::Bf16) ? 1u : 0u;
}

// Byte stride per logical float index for OaLoad/OaStore (matches DTYPE in Storage.slang).
[[nodiscard]] static inline OaU32 OaLoadStoreStrideBytes(
	OaPrecisionTag InTag, OaPrecision InEnginePrecision)
{
	const OaU32 dtype = OaPrecisionDtype(InTag, InEnginePrecision);
	return (dtype == 0u) ? 4u : 2u;
}

// LoadShadersF32 registers pipelines as baseName + "_f32" with DTYPE=0 always.
[[nodiscard]] static inline bool OaPipelineNameIsF32Variant(OaStringView InName)
{
	const OaUsize nameLen = InName.size();
	if (nameLen < 4) return false;
	const char* chars = InName.data();
	const OaUsize base = nameLen - 4;
	return chars[base] == '_' && chars[base + 1] == 'f' && chars[base + 2] == '3' &&
		chars[base + 3] == '2';
}
