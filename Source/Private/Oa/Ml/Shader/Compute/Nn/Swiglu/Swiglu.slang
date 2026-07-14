import Activations;
import Storage;

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

	OaStore(heap[pc.out_idx], idx, Swiglu(OaLoad(heap[pc.gate_idx], idx), OaLoad(heap[pc.up_idx], idx)));
}
