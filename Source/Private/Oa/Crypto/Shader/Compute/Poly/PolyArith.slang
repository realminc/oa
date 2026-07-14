import Modular;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint c_idx;
	uint count;
	uint mode;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + tid.x;
	if (idx >= pc.count * 256) return;

	int result;
	if (pc.mode == 0) {
		result = heap[pc.a_idx].Load<int>(idx * 4) + heap[pc.b_idx].Load<int>(idx * 4);
	} else if (pc.mode == 1) {
		result = heap[pc.a_idx].Load<int>(idx * 4) - heap[pc.b_idx].Load<int>(idx * 4);
	} else if (pc.mode == 2) {
		result = -heap[pc.a_idx].Load<int>(idx * 4);
	} else {
		result = Freeze(Caddq(heap[pc.a_idx].Load<int>(idx * 4)));
	}

	heap[pc.c_idx].Store<int>(idx * 4, result);
}
