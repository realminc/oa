// Core RNG Philox4x32-10 counter-based primitives.

module philox;

static const uint PHILOX_M4X32_0 = 0xD2511F53u;
static const uint PHILOX_M4X32_1 = 0xCD9E8D57u;
static const uint PHILOX_W32_0 = 0x9E3779B9u;
static const uint PHILOX_W32_1 = 0xBB67AE85u;

public void multiply_high_low(uint left, uint right, out uint high, out uint low) {
	// Exact unsigned 32x32 -> 64 product, represented by two 32-bit words.
	// The donor uses uint64_t here; limb products retain the same Philox bits
	// on devices without shaderInt64. Each intermediate sum fits in uint32.
	uint left_low = left & 0xFFFFu;
	uint left_high = left >> 16u;
	uint right_low = right & 0xFFFFu;
	uint right_high = right >> 16u;
	uint product_low = left_low * right_low;
	uint middle0 = left_high * right_low + (product_low >> 16u);
	uint middle1 = left_low * right_high + (middle0 & 0xFFFFu);
	high = left_high * right_high + (middle0 >> 16u) + (middle1 >> 16u);
	low = (middle1 << 16u) | (product_low & 0xFFFFu);
}

public void philox_round(inout uint4 state, inout uint2 key) {
	uint high0;
	uint low0;
	uint high1;
	uint low1;
	multiply_high_low(PHILOX_M4X32_0, state.x, high0, low0);
	multiply_high_low(PHILOX_M4X32_1, state.z, high1, low1);
	state.x = high1 ^ state.y ^ key.x;
	state.y = low1;
	state.z = high0 ^ state.w ^ key.y;
	state.w = low0;
	key.x += PHILOX_W32_0;
	key.y += PHILOX_W32_1;
}

public uint4 philox_rng(uint4 counter, uint2 key) {
	uint4 state = counter;
	uint2 round_key = key;
	[unroll] for (uint round = 0u; round < 10u; ++round) {
		philox_round(state, round_key);
	}
	return state;
}

public float philox_to_float01(uint bits) {
	return float(bits) * 2.3283064365386963e-10f;
}

public void philox_to_normal(uint4 bits, out float first, out float second) {
	float uniform0 = philox_to_float01(bits.x);
	float uniform1 = philox_to_float01(bits.y);
	float radius = sqrt(-2.0f * log(uniform0 + 1.0e-10f));
	float angle = 6.28318530718f * uniform1;
	first = radius * cos(angle);
	second = radius * sin(angle);
}
