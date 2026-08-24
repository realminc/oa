import activations;
import storage;

struct PushConstants {
	uint gate_idx;
	uint up_idx;
	uint out_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	oaStore(heap[pc.out_idx], idx, Swiglu(oaLoad(heap[pc.gate_idx], idx), oaLoad(heap[pc.up_idx], idx)));
}
