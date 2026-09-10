// Core RNG Philox4x32-10 counter-based primitives.

module philox;

static const uint PHILOX_M4X32_0 = 0xD2511F53u;
static const uint PHILOX_M4X32_1 = 0xCD9E8D57u;
static const uint PHILOX_W32_0 = 0x9E3779B9u;
static const uint PHILOX_W32_1 = 0xBB67AE85u;

public void multiply_high_low(uint left, uint right, out uint high, out uint low) {
	uint64_t product = uint64_t(left) * uint64_t(right);
	high = uint(product >> 32u);
	low = uint(product & 0xFFFFFFFFu);
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
