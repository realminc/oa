// OA donor Keccak-f[1600], adapted to uint2 lanes so the kernel does not
// require Vulkan shaderInt64.

static const uint RC_LO[24] = {
	0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
	0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
	0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
	0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
	0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
	0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
};

static const uint RC_HI[24] = {
	0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
	0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
	0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
	0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
	0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
	0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
};

static const uint ROT[25] = {
	0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
	25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14
};

uint2 rotl64(uint2 value, uint shift) {
	if (shift == 0) return value;
	if (shift >= 32) {
		value = uint2(value.y, value.x);
		shift -= 32;
	}
	if (shift == 0) return value;
	return uint2(
		(value.x << shift) | (value.y >> (32 - shift)),
		(value.y << shift) | (value.x >> (32 - shift))
	);
}

void keccak_f1600_u32(inout uint2 state[25]) {
	for (uint round = 0; round < 24; ++round) {
		uint2 columns[5];
		for (uint i = 0; i < 5; ++i) {
			columns[i] = state[i] ^ state[5 + i] ^ state[10 + i] ^ state[15 + i] ^ state[20 + i];
		}
		uint2 differences[5];
		for (uint i = 0; i < 5; ++i) {
			differences[i] = columns[(i + 4) % 5] ^ rotl64(columns[(i + 1) % 5], 1);
		}
		for (uint i = 0; i < 25; ++i) state[i] ^= differences[i % 5];

		uint2 rotated[25];
		for (uint i = 0; i < 25; ++i) {
			uint x = i % 5;
			uint y = i / 5;
			rotated[5 * ((2 * x + 3 * y) % 5) + y] = rotl64(state[i], ROT[i]);
		}
		for (uint y = 0; y < 5; ++y) {
			uint base = y * 5;
			state[base] = rotated[base] ^ (~rotated[base + 1] & rotated[base + 2]);
			state[base + 1] = rotated[base + 1] ^ (~rotated[base + 2] & rotated[base + 3]);
			state[base + 2] = rotated[base + 2] ^ (~rotated[base + 3] & rotated[base + 4]);
			state[base + 3] = rotated[base + 3] ^ (~rotated[base + 4] & rotated[base]);
			state[base + 4] = rotated[base + 4] ^ (~rotated[base] & rotated[base + 1]);
		}
		state[0] ^= uint2(RC_LO[round], RC_HI[round]);
	}
}
