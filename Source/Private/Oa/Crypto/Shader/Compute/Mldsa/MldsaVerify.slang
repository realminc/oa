// ML-DSA-65 batch verification — one workgroup per verification.
//
// Verifies a signature against a message and public key.
// This is the critical path for block validation (many tx sigs per block).
//
// Input: packed buffers of pubkeys, signatures, messages (uniform length)
// Output: results buffer (1 = valid, 0 = invalid)
//
// ML-DSA-65 params: k=6, l=5, n=256, q=8380417, gamma1=2^19, gamma2=(q-1)/32,
//                   beta=196, tau=49, d=13
//
// Verify algorithm:
//   1. Decode pk -> (rho, t1[k])
//   2. Decode sig -> (c_tilde, z[l], h)
//   3. Check ||z||_inf < gamma1 - beta
//   4. A = ExpandA(rho)              // k*l SHAKE-128 calls
//   5. c = SampleInBall(c_tilde)     // SHAKE-256 -> sparse poly
//   6. w1' = A*NTT(z) - NTT(c)*NTT(t1 << d)
//   7. w1' = UseHint(h, w1')
//   8. c_tilde' = H(tr || msg || w1')  // SHAKE-256
//   9. Accept iff c_tilde == c_tilde'
//
// Strategy: one workgroup with 256 threads handles one complete verification.
// All NTT/SHAKE/polynomial ops are inlined using groupshared memory.
// No global memory round-trips between steps.

struct PushConstants {
	uint pubkeys_idx;
	uint signatures_idx;
	uint messages_idx;
	uint results_idx;
	uint count;
	uint msg_len;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const int Q = 8380417;
static const int QINV = 58728449;
static const uint K = 6;
static const uint L = 5;
static const uint N = 256;
static const uint D = 13;
static const int GAMMA1 = 524288;
static const int GAMMA2 = 261888;
static const int BETA = 196;
static const uint TAU = 49;
static const uint PUBKEY_BYTES = 1952;
static const uint SIG_BYTES = 3309;

static const int ZETAS[256] = {
	0, 25847, -2608894, -518909, 237124, -777960, -876248, 466468,
	1826347, 2353451, -359251, -2091905, 3119733, -2884855, 3111497, 2680103,
	2725464, 1024112, -1079900, 3585928, -549488, -1119584, 2619752, -2108549,
	-2118186, -3859737, -1399561, -3277672, 1757237, -19422, 4010497, 280005,
	2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
	-3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299,
	-1699267, -1643818, 3505694, -3821735, 3507263, -2140649, -1600420, 3699596,
	811944, 531354, 954230, 3881043, 3900724, -2556880, 2071892, -2797779,
	-3930395, -1528703, -3677745, -3041255, -1452451, 3475950, 2176455, -1585221,
	-1257611, 1939314, -4083598, -1000202, -3190144, -3157330, -3632928, 126922,
	3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
	-671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430,
	-3343383, 264944, 508951, 3097992, 44288, -1100098, 904516, 3958618,
	-3724342, -8578, 1653064, -3249728, 2389356, -210977, 759969, -1316856,
	189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589, 1341330,
	1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961,
	2091667, 3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462,
	266997, 2434439, -1235728, 3513181, -3520352, -3759364, -1197226, -3193378,
	900702, 1859098, 909542, 819034, 495491, -1613174, -43260, -522500,
	-655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622, -3595838,
	342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044,
	2842341, 2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974,
	-3767016, 1250494, 2635921, -3548272, -2994039, 1869119, 1903435, -1050970,
	-1333058, 1237275, -3318210, -1430225, -451100, 1312455, 3306115, -1962642,
	-1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880, 3406031,
	-542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993,
	-2013608, 2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385,
	-3183426, 162844, 1616392, 3014001, 810149, 1652634, -3694233, -1799107,
	-3038916, 3523897, 3866901, 269760, 2213111, -975884, 1717735, 472078,
	-426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646, -3833893,
	-2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687,
	-554416, 3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782
};

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

int montgomery_reduce(int64_t InA) {
	int t = int(InA) * QINV;
	return int((InA - int64_t(t) * int64_t(Q)) >> 32);
}

int caddq(int InA) {
	InA += (InA >> 31) & Q;
	return InA;
}

int freeze(int InA) {
	InA = caddq(InA);
	InA -= Q;
	InA += (InA >> 31) & Q;
	return InA;
}

uint64_t rotl64(uint64_t InVal, uint InShift) {
	if (InShift == 0) return InVal;
	return (InVal << InShift) | (InVal >> (64 - InShift));
}

void keccak_f1600(inout uint64_t s[25]) {
	for (uint round = 0; round < 24; round++) {
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

		for (uint i = 0; i < 25; i++) s[i] ^= d[i % 5];

		uint64_t t[25];
		for (uint i = 0; i < 25; i++) {
			uint x = i % 5;
			uint y = i / 5;
			t[5 * ((2 * x + 3 * y) % 5) + y] = rotl64(s[i], KECCAK_ROT[i]);
		}

		for (uint y = 0; y < 5; y++) {
			uint b = y * 5;
			s[b + 0] = t[b + 0] ^ (~t[b + 1] & t[b + 2]);
			s[b + 1] = t[b + 1] ^ (~t[b + 2] & t[b + 3]);
			s[b + 2] = t[b + 2] ^ (~t[b + 3] & t[b + 4]);
			s[b + 3] = t[b + 3] ^ (~t[b + 4] & t[b + 0]);
			s[b + 4] = t[b + 4] ^ (~t[b + 0] & t[b + 1]);
		}

		s[0] ^= KECCAK_RC[round];
	}
}

// NTT in groupshared memory (256 threads cooperate)
groupshared int s_ntt[256];

void ntt_shared(uint InLane) {
	for (uint stage = 0; stage < 8; stage++) {
		uint len = 128u >> stage;
		uint block_size = 2 * len;
		uint group_idx = InLane / block_size;
		uint pos = InLane % block_size;

		if (pos < len) {
			uint zeta_idx = (1u << stage) + group_idx;
			int zeta = ZETAS[zeta_idx];
			uint j_lo = group_idx * block_size + pos;
			uint j_hi = j_lo + len;

			int a_lo = s_ntt[j_lo];
			int a_hi = s_ntt[j_hi];
			int t = montgomery_reduce(int64_t(zeta) * int64_t(a_hi));

			s_ntt[j_lo] = a_lo + t;
			s_ntt[j_hi] = a_lo - t;
		}
		GroupMemoryBarrierWithGroupSync();
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
// ============================================================================
// WIP / ROADMAP — NOT A REAL VERIFY. Steps 1-8 above are not implemented; the
// NTT/Poly/SHAKE building blocks below are real, but decode/challenge/hint/
// comparison are missing. This stub is intentionally FAIL-CLOSED: it writes 0
// (INVALID) for every input, so wiring it by mistake rejects all signatures
// rather than accepting them. Do NOT expose this through a public API until the
// full algorithm lands. Until then, ML-DSA verification runs on the CPU (liboqs,
// Source/Private/Oa/Crypto/Sign.cpp).
// ============================================================================
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint verify_idx = gid.x;
	if (verify_idx >= pc.count) return;

	uint lane = tid.x;

	s_ntt[lane] = int(lane);
	GroupMemoryBarrierWithGroupSync();
	ntt_shared(lane);

	if (lane == 0) {
		// Fail-closed: unimplemented verify must never report "valid".
		heap[pc.results_idx].Store<uint>(verify_idx * 4, 0);
	}
}
