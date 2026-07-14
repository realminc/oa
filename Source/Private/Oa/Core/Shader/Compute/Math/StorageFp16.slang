// StorageFp16.slang — Pure FP16 storage (2 bytes/element)
// No DTYPE branching. Always uses float16 layout.
//
// FP16 values stored as 16-bit half-floats via uint16 bit pattern.
// OaLoad: read 2 bytes, extend to float32
// OaStore: truncate float32 to float16, write 2 bytes
//
// NOTE: Stride is 2 bytes per element index.
// The caller is responsible for knowing that this module is active.
// Do NOT use with buffers allocated with FP32/4-byte stride.

module StorageFp16;

// --- helpers ---

// Float32 → Float16 (truncation, round-to-nearest-even)
public uint F32ToF16(float val) {
	uint bits = asuint(val);
	uint sign = (bits >> 16u) & 0x8000u;
	uint exp  = (bits >> 23u) & 0xFFu;
	uint mant = bits & 0x7FFFFFu;

	if (exp == 0xFFu) {
		// NaN or Inf — preserve as FP16 inf/nan
		return sign | 0x7C00u | (mant >> 13u);
	}

	int exp_i = int(exp) - 127 + 15;
	if (exp_i >= 31) {
		return sign | 0x7C00u; // Inf
	}
	if (exp_i <= 0) {
		// Subnormal or zero
		if (exp_i < -10) return sign;
		mant = (mant | 0x800000u) >> (1 - exp_i);
		return sign | (mant >> 13u);
	}

	return sign | (uint(exp_i) << 10u) | (mant >> 13u);
}

// Float16 → Float32 (zero-extend, exact)
public float F16ToF32(uint f16) {
	uint sign = (f16 & 0x8000u) << 16u;
	uint exp  = (f16 & 0x7C00u) >> 10u;
	uint mant = f16 & 0x03FFu;

	if (exp == 0u) {
		if (mant == 0u) return asfloat(sign); // ±0
		// Subnormal: normalize
		uint n = 0xFFFFu;
		uint m = mant;
		while (m >> 10u) == 0u { m <<= 1u; n--; }
		exp = 1u + 127u - 15u + n;
		mant = m & 0x03FFu;
	} else if (exp == 31u) {
		exp = 0xFFu; // NaN/Inf
	} else {
		exp = exp + 127u - 15u;
	}

	return asfloat((sign) | (exp << 23u) | (mant << 13u));
}

// --- public API ---

// Load element at logical index `idx` from `buf`, returning float32
[ForceInline]
public float OaLoad(ByteAddressBuffer buf, uint idx) {
	// FP16 values stored at 2-byte stride
	uint byte_off = idx * 2u;
	uint word = buf.Load(byte_off & ~3u);
	uint shift = (byte_off & 2u) * 8u;
	uint f16 = (word >> shift) & 0xFFFFu;
	return F16ToF32(f16);
}

[ForceInline]
public float OaLoad(RWByteAddressBuffer buf, uint idx) {
	uint byte_off = idx * 2u;
	uint word = buf.Load(byte_off & ~3u);
	uint shift = (byte_off & 2u) * 8u;
	uint f16 = (word >> shift) & 0xFFFFu;
	return F16ToF32(f16);
}

// Store float32 `val` at logical index `idx` into `buf` as FP16
[ForceInline]
public void OaStore(RWByteAddressBuffer buf, uint idx, float val) {
	uint f16 = F32ToF16(val);
	uint byte_off = idx * 2u;
	uint word_off = byte_off & ~3u;
	uint shift = (byte_off & 2u) * 8u;
	uint mask = 0xFFFFu << shift;
	uint old_word = buf.Load(word_off);
	uint new_word = (old_word & ~mask) | (f16 << shift);
	buf.Store(word_off, new_word);
}

// Always-FP32 variants (for buffer metadata blocks, scalars, etc.)
[ForceInline]
public float OaLoadF32(ByteAddressBuffer buf, uint idx) {
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public float OaLoadF32(RWByteAddressBuffer buf, uint idx) {
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public void OaStoreF32(RWByteAddressBuffer buf, uint idx, float val) {
	buf.Store(idx * 4u, asuint(val));
}

// Atomic add — always FP32 (atomic ops require native 32-bit)
[ForceInline]
public void OaAtomicAdd(RWByteAddressBuffer buf, uint idx, float value) {
	uint addr = idx * 4u;
	uint oldVal = 0u;
	uint assumed;
	do {
		assumed = oldVal;
		uint newVal = asuint(asfloat(assumed) + value);
		buf.InterlockedCompareExchange(addr, assumed, newVal, oldVal);
	} while (assumed != oldVal);
}