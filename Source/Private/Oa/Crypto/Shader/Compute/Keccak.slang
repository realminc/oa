// Keccak.slang — Shared Keccak-f[1600] implementation for import.
//
// Provides KeccakF1600() function that can be imported by other kernels.
// Uses uint64_t for simplicity (GPU native support).

static const uint64_t KECCAK_RC[24] = {
	0x0000000000000001ULL, 0x0000000000008082ULL,
	0x800000000000808aULL, 0x8000000080008000ULL,
	0x000000000000808bULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008aULL, 0x0000000000000088ULL,
	0x0000000080008009ULL, 0x000000008000000aULL,
	0x000000008000808bULL, 0x800000000000008bULL,
	0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800aULL, 0x800000008000000aULL,
	0x8000000080008081ULL, 0x8000000000008080ULL,
	0x0000000080000001ULL, 0x8000000080008008ULL
};

static const uint KECCAK_ROT[25] = {
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14
};

uint64_t rotl64(uint64_t InVal, uint InShift) {
	if (InShift == 0) return InVal;
	return (InVal << InShift) | (InVal >> (64 - InShift));
}

void KeccakF1600(inout uint64_t s[25]) {
	for (uint round = 0; round < 24; round++) {
		// Theta
		uint64_t c[5];
		c[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		c[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		c[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		c[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		c[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		uint64_t d[5];
		d[0] = c[4] ^ rotl64(c[1], 1);
		d[1] = c[0] ^ rotl64(c[2], 1);
		d[2] = c[1] ^ rotl64(c[3], 1);
		d[3] = c[2] ^ rotl64(c[4], 1);
		d[4] = c[3] ^ rotl64(c[0], 1);

		for (uint i = 0; i < 25; i++) {
			s[i] ^= d[i % 5];
		}

		// Rho + Pi (combined)
		uint64_t t[25];
		for (uint i = 0; i < 25; i++) {
			uint x = i % 5;
			uint y = i / 5;
			uint nx = y;
			uint ny = (2 * x + 3 * y) % 5;
			t[5 * ny + nx] = rotl64(s[i], KECCAK_ROT[i]);
		}

		// Chi
		for (uint y = 0; y < 5; y++) {
			uint b = y * 5;
			s[b + 0] = t[b + 0] ^ (~t[b + 1] & t[b + 2]);
			s[b + 1] = t[b + 1] ^ (~t[b + 2] & t[b + 3]);
			s[b + 2] = t[b + 2] ^ (~t[b + 3] & t[b + 4]);
			s[b + 3] = t[b + 3] ^ (~t[b + 4] & t[b + 0]);
			s[b + 4] = t[b + 4] ^ (~t[b + 0] & t[b + 1]);
		}

		// Iota
		s[0] ^= KECCAK_RC[round];
	}
}