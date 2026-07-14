import Modular;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint c_idx;
	uint count;
	uint accumulate;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count * 256) return;

	int av = heap[pc.a_idx].Load<int>(idx * 4);
	int bv = heap[pc.b_idx].Load<int>(idx * 4);
	uint prod_lo = uint(av) * uint(bv);
	int prod_hi = Imulhi(av, bv);
	int prod = MontgomeryReduce32(prod_lo, prod_hi);

	if (pc.accumulate != 0) {
		heap[pc.c_idx].Store<int>(idx * 4, heap[pc.c_idx].Load<int>(idx * 4) + prod);
	} else {
		heap[pc.c_idx].Store<int>(idx * 4, prod);
	}
}
