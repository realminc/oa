// Storage — DTYPE-specialized storage helpers (the canonical dtype-aware module).
//
// OaLoad/OaStore branch on the DTYPE spec constant (constant_id 0): DTYPE=0 →
// 4-byte fp32, DTYPE=1 → native 2-byte bf16 (paired-half CAS store, see OaStore).
// Because DTYPE is a *specialization* constant, the branch is resolved at pipeline
// creation and the unused path is dead-code-eliminated — there is NO runtime branch
// cost. This is exactly what lets ONE kernel (GEMM / Linear / epilogue / elementwise)
// serve both fp32 and bf16, selected per-dispatch via SetLastNodeDtype.
//
// Use this module for any kernel that must handle both precisions. The single-dtype
// modules are the specializations, for shaders that provably see one precision only:
//   - StorageFp32  — fp32-only (e.g. gradient accumulators that must stay fp32)
//   - StorageBf16  — bf16-only
//   - StorageMixed — mixed fp32/bf16 (optimizer master+copy, cast kernels)

module Storage;

[[vk::constant_id(0)]] const uint DTYPE = 0;

// Public accessor so importing modules can specialize their own load/store WIDTH on
// the storage dtype (the spec constant itself is module-private). Used by GemmCmSgBf16
// to pick a 128-bit uint4 bf16 staging path vs the fp32 float4 path.
[ForceInline]
public bool OaIsBf16() { return DTYPE == 1u; }

[ForceInline]
public float OaLoad(ByteAddressBuffer buf, uint idx) {
	if (DTYPE == 1u) {
		// BF16: Load 16-bit value using bit manipulation (no uint16_t type needed)
		uint byte_off = idx * 2u;
		uint word = buf.Load(byte_off & ~3u);
		uint shift = (byte_off & 2u) * 8u;
		uint bf16 = (word >> shift) & 0xFFFFu;
		return asfloat(bf16 << 16u);
	}
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public float OaLoad(RWByteAddressBuffer buf, uint idx) {
	if (DTYPE == 1u) {
		// BF16: Load 16-bit value using bit manipulation (no uint16_t type needed)
		uint byte_off = idx * 2u;
		uint word = buf.Load(byte_off & ~3u);
		uint shift = (byte_off & 2u) * 8u;
		uint bf16 = (word >> shift) & 0xFFFFu;
		return asfloat(bf16 << 16u);
	}
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public void OaStore(RWByteAddressBuffer buf, uint idx, float val) {
	if (DTYPE == 1u) {
		// BF16: Store 16-bit value using bit manipulation (no uint16_t type needed)
		uint bits = asuint(val);
		uint bf16 = (bits >> 16u) & 0xFFFFu;
		uint byte_off = idx * 2u;
		uint word_off = byte_off & ~3u;
		uint shift = (byte_off & 2u) * 8u;
		uint mask = 0xFFFFu << shift;
		// ATOMIC read-modify-write of the 16-bit half. A plain Load/Store RMW races:
		// two threads writing the two bf16 halves of the same 32-bit word both read
		// the old word and the last store wins, silently dropping the other half's
		// write. Because parallel kernels write adjacent (paired) elements from
		// adjacent lanes, ~half of every bf16 elementwise/gather/scan output was
		// lost — collapsing bf16 forward activations (~half magnitude) and, through
		// recurrence, the whole conditional signal. CAS makes each half-word update
		// atomic. Mirrors OaAtomicAdd below.
		uint oldVal = buf.Load(word_off);
		uint assumed;
		do {
			assumed = oldVal;
			uint new_word = (assumed & ~mask) | (bf16 << shift);
			buf.InterlockedCompareExchange(word_off, assumed, new_word, oldVal);
		} while (assumed != oldVal);
		return;
	}
	buf.Store(idx * 4u, asuint(val));
}

[ForceInline]
public float OaLoadF32(RWByteAddressBuffer buf, uint idx) {
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public float OaLoadF32(ByteAddressBuffer buf, uint idx) {
	return asfloat(buf.Load(idx * 4u));
}

[ForceInline]
public void OaStoreF32(RWByteAddressBuffer buf, uint idx, float val) {
	buf.Store(idx * 4u, asuint(val));
}

// Atomic add for float32 via CAS loop (always FP32 regardless of DTYPE). This is
// deliberate: gradient accumulators MUST be fp32 for numerical soundness, and in a
// mixed-precision dispatch (bf16 inputs, fp32 grad accumulator) the atomic target
// is fp32 even when the pipeline's DTYPE spec constant is 1 (bf16). Callers that
// atomic-accumulate therefore allocate the target buffer as fp32 — e.g. GRU/RNN
// scan backward, and the Mamba3/Empyrealm preprocess backward (whose dt_bias grad
// buffer is fp32 because the whole SSM math runs as an fp32 island). Branching this
// on DTYPE would corrupt those fp32 accumulators — do not.
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
