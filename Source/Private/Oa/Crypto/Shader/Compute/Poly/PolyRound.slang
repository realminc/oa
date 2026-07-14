import Modular;

struct PushConstants {
	uint input_idx;
	uint out_hi_idx;
	uint out_lo_idx;
	uint count;
	uint mode;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const int D = 13;
static const int GAMMA2 = 261888;

void power2round(int InA, out int OutHi, out int OutLo) {
	InA = Caddq(InA);
	OutHi = (InA + (1 << (D - 1)) - 1) >> D;
	OutLo = InA - (OutHi << D);
}

void decompose(int InA, out int OutHi, out int OutLo) {
	InA = Caddq(InA);
	OutLo = InA % (2 * GAMMA2);
	// Center
	OutLo -= GAMMA2;
	// Correction for edge case
	int mask = (GAMMA2 - 1 - OutLo) >> 31;
	OutLo -= mask & 1;

	if (InA - OutLo == Q - 1) {
		OutHi = 0;
		OutLo = OutLo - 1;
	} else {
		OutHi = (InA - OutLo) / (2 * GAMMA2);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count * 256) return;

	int a = heap[pc.input_idx].Load<int>(idx * 4);
	int hi, lo;

	if (pc.mode == 0) {
		power2round(a, hi, lo);
		heap[pc.out_hi_idx].Store<int>(idx * 4, hi);
		heap[pc.out_lo_idx].Store<int>(idx * 4, lo);
	} else if (pc.mode == 1) {
		decompose(a, hi, lo);
		heap[pc.out_hi_idx].Store<int>(idx * 4, hi);
		heap[pc.out_lo_idx].Store<int>(idx * 4, lo);
	} else if (pc.mode == 2) {
		decompose(a, hi, lo);
		heap[pc.out_hi_idx].Store<int>(idx * 4, hi);
	} else {
		decompose(a, hi, lo);
		heap[pc.out_lo_idx].Store<int>(idx * 4, lo);
	}
}
