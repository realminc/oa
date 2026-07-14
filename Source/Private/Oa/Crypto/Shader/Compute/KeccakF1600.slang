// Keccak-f[1600] permutation — batch GPU implementation.
//
// State: 25 x 64-bit lanes (200 bytes), stored as 50 x uint32 (lo/hi pairs).
// 24 rounds of theta/rho/pi/chi/iota.
//
// Pure 32-bit arithmetic — no uint64_t. Each 64-bit lane is a uint2 (lo, hi).
// Buffer layout: states[idx*50 + 2*lane] = lo, states[idx*50 + 2*lane+1] = hi.
// Byte-compatible with uint64_t[25] on little-endian GPUs.
//
// One thread per state. Workgroup of 256 = 256 independent permutations.
// Dispatch ceil(N / 256) workgroups for N states.
//
// Out-of-place: reads state from in_idx, writes the permuted state to out_idx
// (in and out may alias for an in-place permutation).

struct PushConstants {
	uint in_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Round constants split into lo/hi 32-bit halves.
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

// Rho rotation offsets indexed as [5*y + x]
static const uint ROT[25] = {
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14
};

// 64-bit left rotation on (lo, hi) pair.
uint2 rotl64(uint2 v, uint n) {
	if (n == 0) return v;
	if (n >= 32) {
		v = uint2(v.y, v.x);
		n -= 32;
	}
	if (n == 0) return v;
	return uint2(
		(v.x << n) | (v.y >> (32 - n)),
		(v.y << n) | (v.x >> (32 - n))
	);
}

void keccak_f1600(inout uint2 s[25]) {
	for (uint round = 0; round < 24; round++) {
		// Theta
		uint2 c[5];
		for (uint i = 0; i < 5; i++) {
			c[i] = uint2(
				s[i].x ^ s[5+i].x ^ s[10+i].x ^ s[15+i].x ^ s[20+i].x,
				s[i].y ^ s[5+i].y ^ s[10+i].y ^ s[15+i].y ^ s[20+i].y
			);
		}

		uint2 d[5];
		d[0] = uint2(c[4].x ^ rotl64(c[1], 1).x, c[4].y ^ rotl64(c[1], 1).y);
		d[1] = uint2(c[0].x ^ rotl64(c[2], 1).x, c[0].y ^ rotl64(c[2], 1).y);
		d[2] = uint2(c[1].x ^ rotl64(c[3], 1).x, c[1].y ^ rotl64(c[3], 1).y);
		d[3] = uint2(c[2].x ^ rotl64(c[4], 1).x, c[2].y ^ rotl64(c[4], 1).y);
		d[4] = uint2(c[3].x ^ rotl64(c[0], 1).x, c[3].y ^ rotl64(c[0], 1).y);

		for (uint i = 0; i < 25; i++) {
			uint di = i % 5;
			s[i] = uint2(s[i].x ^ d[di].x, s[i].y ^ d[di].y);
		}

		// Rho + Pi (combined)
		uint2 t[25];
		for (uint i = 0; i < 25; i++) {
			uint x = i % 5;
			uint y = i / 5;
			uint nx = y;
			uint ny = (2 * x + 3 * y) % 5;
			t[5 * ny + nx] = rotl64(s[i], ROT[i]);
		}

		// Chi
		for (uint y = 0; y < 5; y++) {
			uint b = y * 5;
			s[b+0] = uint2(t[b+0].x ^ (~t[b+1].x & t[b+2].x), t[b+0].y ^ (~t[b+1].y & t[b+2].y));
			s[b+1] = uint2(t[b+1].x ^ (~t[b+2].x & t[b+3].x), t[b+1].y ^ (~t[b+2].y & t[b+3].y));
			s[b+2] = uint2(t[b+2].x ^ (~t[b+3].x & t[b+4].x), t[b+2].y ^ (~t[b+3].y & t[b+4].y));
			s[b+3] = uint2(t[b+3].x ^ (~t[b+4].x & t[b+0].x), t[b+3].y ^ (~t[b+4].y & t[b+0].y));
			s[b+4] = uint2(t[b+4].x ^ (~t[b+0].x & t[b+1].x), t[b+4].y ^ (~t[b+0].y & t[b+1].y));
		}

		// Iota
		s[0] = uint2(s[0].x ^ RC_LO[round], s[0].y ^ RC_HI[round]);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count) return;

	uint base = idx * 50;
	uint2 s[25];
	for (uint i = 0; i < 25; i++) {
		s[i] = uint2(
			heap[pc.in_idx].Load<uint>((base + 2*i) * 4),
			heap[pc.in_idx].Load<uint>((base + 2*i + 1) * 4)
		);
	}

	keccak_f1600(s);

	for (uint i = 0; i < 25; i++) {
		heap[pc.out_idx].Store<uint>((base + 2*i) * 4, s[i].x);
		heap[pc.out_idx].Store<uint>((base + 2*i + 1) * 4, s[i].y);
	}
}
