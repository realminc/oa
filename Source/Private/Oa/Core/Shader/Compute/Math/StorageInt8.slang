// StorageInt8.slang — Pure INT8 storage (1 byte/element)
// No DTYPE branching. Always uses signed 8-bit integer layout.
//
// Values stored as int8 at 1 byte per element index.
// OaLoad: read 1 byte, sign-extend to int32, convert to float32
// OaStore: clamp float32 to [-128, 127], write as int8
//
// NOTE: Stride is 1 byte per element index.
// This is for WEIGHT-ONLY quantization. Activations/gradients are FP32.
// The weight matrix is quantized offline; at runtime we dequantize-on-load
// into FP32 registers, compute in FP32, and optionally re-quantize on store.
//
// For INT8×INT8 dot product GEMM, use GemmInt8.slang directly.

module StorageInt8;

// --- helpers ---

// Clamp float to int8 range
public int ClampToInt8(float val) {
	if (val > 127.0f) return 127;
	if (val < -128.0f) return -128;
	return int(val);
}

// --- public API ---

// Load element at logical index `idx` from `buf`, returning float32
// ByteAddressBuffer.Load(byteOffset) returns uint. We cast to int for sign extension.
[ForceInline]
public float OaLoad(ByteAddressBuffer buf, uint idx) {
	// Load single byte, sign-extend: the uint MSB is the sign bit
	uint raw = buf.Load(idx);  // Load returns byte at offset idx (scalar uint from byte access)
	// Sign-extend: if bit 7 is set, OR with 0xFFFFFF00
	int val = (raw & 0x80u) != 0u ? int(raw | 0xFFFFFF00u) : int(raw);
	return float(val);
}

[ForceInline]
public float OaLoad(RWByteAddressBuffer buf, uint idx) {
	uint raw = buf.Load(idx);
	int val = (raw & 0x80u) != 0u ? int(raw | 0xFFFFFF00u) : int(raw);
	return float(val);
}

// Store float32 `val` at logical index `idx` into `buf` as INT8
[ForceInline]
public void OaStore(RWByteAddressBuffer buf, uint idx, float val) {
	int quantized = ClampToInt8(val);
	buf.Store(idx, uint(quantized & 0xFFu));
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