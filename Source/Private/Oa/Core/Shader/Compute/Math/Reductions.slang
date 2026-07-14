// oa::reductions — Shared reduction primitives for Slang shaders
//
// Import: `import Reductions;`
// Wave-intrinsic reductions used by RMSNorm, Softmax, LayerNorm, CrossEntropy.
// Pure math — no bindings, no push constants, no entry points.
// Buffer I/O via oa.slang (BF16/FP32 transparent).
//
// groupshared memory is owned by this module. Callers do NOT declare their own
// shared buffer for reductions — they call ReduceSum256/ReduceMax256/etc. which
// handle the full load→reduce→return pipeline internally.
//
// Full 256-wide groupshared tree reduction. A previous "optimized" variant used
// WaveActiveSum/WaveActiveMax for the intra-wave step plus an 8-slot inter-wave
// combine, but the inter-wave step assumed the block's wave sums all landed in the
// first wave (waveCount <= WaveGetLaneCount()) and combined them with a second
// WaveActiveSum. On devices/drivers where that assumption or the reported lane
// count did not hold, the combine silently dropped half the partial sums —
// RMSNorm's inv_rms came out a clean sqrt(2) too large. The tree reduction below
// is subgroup-size-agnostic and matches the proven-correct local reductions in
// Softmax/LayerNorm. All 256 threads of the group MUST call these (no divergence).

module Reductions;

// Reductions always compute in FP32 — OaLoad returns float regardless of
// storage format. DTYPE-aware Storage ensures correct byte stride (2-byte
// BF16, 4-byte FP32) when reading from BF16-stored buffers.
import Storage;

groupshared float g_reduce_buf[256];

public float ReduceSum256(float InLocalVal, uint InLane) {
	g_reduce_buf[InLane] = InLocalVal;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (InLane < s) g_reduce_buf[InLane] += g_reduce_buf[InLane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	return g_reduce_buf[0];
}

public float ReduceMax256(float InLocalVal, uint InLane) {
	g_reduce_buf[InLane] = InLocalVal;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (InLane < s) g_reduce_buf[InLane] = max(g_reduce_buf[InLane], g_reduce_buf[InLane + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	return g_reduce_buf[0];
}

[ForceInline]
public float InvRms(
	RWByteAddressBuffer InX, uint InBase, uint InCols, float InEps, uint InLane
) {
	float localSq = 0.0;
	for (uint i = InLane; i < InCols; i += 256) {
		float v = OaLoad(InX, InBase + i);
		localSq += v * v;
	}
	float sumSq = ReduceSum256(localSq, InLane);
	return 1.0 / sqrt(sumSq / float(InCols) + InEps);
}

[ForceInline]
public float RowMax(
	RWByteAddressBuffer InX, uint InBase, uint InCols, uint InLane
) {
	float localMax = -3.402823466e+38f;
	for (uint i = InLane; i < InCols; i += 256) {
		localMax = max(localMax, OaLoad(InX, InBase + i));
	}
	return ReduceMax256(localMax, InLane);
}

[ForceInline]
public float RowSumExp(
	RWByteAddressBuffer InX, uint InBase, uint InCols, float InOffset, uint InLane
) {
	float localSum = 0.0;
	for (uint i = InLane; i < InCols; i += 256) {
		localSum += exp(OaLoad(InX, InBase + i) - InOffset);
	}
	return ReduceSum256(localSum, InLane);
}
