import Modular;

struct PushConstants {
	uint a0_idx;
	uint a1_idx;
	uint output_idx;
	uint count;
	uint mode;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

static const int GAMMA2 = 261888;

int highbits(int InA) {
	InA = Caddq(InA);
	int a0 = InA % (2 * GAMMA2);
	a0 -= GAMMA2;
	int mask = (GAMMA2 - 1 - a0) >> 31;
	a0 -= mask & 1;

	if (InA - a0 == Q - 1) {
		return 0;
	}
	return (InA - a0) / (2 * GAMMA2);
}

int make_hint(int InA0, int InA1) {
	int h0 = highbits(InA0);
	int h1 = highbits(InA0 + InA1);
	return (h0 != h1) ? 1 : 0;
}

int use_hint(int InA, int InHint) {
	InA = Caddq(InA);
	int a0 = InA % (2 * GAMMA2);
	a0 -= GAMMA2;
	int mask = (GAMMA2 - 1 - a0) >> 31;
	a0 -= mask & 1;

	int a1;
	if (InA - a0 == Q - 1) {
		a1 = 0;
		a0 = a0 - 1;
	} else {
		a1 = (InA - a0) / (2 * GAMMA2);
	}

	if (InHint == 0) return a1;

	if (a0 > 0) {
		int m = (Q - 1) / (2 * GAMMA2);
		return (a1 + 1) % (m + 1);
	} else {
		int m = (Q - 1) / (2 * GAMMA2);
		return (a1 + m) % (m + 1);
	}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count * 256) return;

	if (pc.mode == 0) {
		heap[pc.output_idx].Store<int>(idx * 4, make_hint(
			heap[pc.a0_idx].Load<int>(idx * 4),
			heap[pc.a1_idx].Load<int>(idx * 4)
		));
	} else {
		heap[pc.output_idx].Store<int>(idx * 4, use_hint(
			heap[pc.a0_idx].Load<int>(idx * 4),
			heap[pc.a1_idx].Load<int>(idx * 4)
		));
	}
}
