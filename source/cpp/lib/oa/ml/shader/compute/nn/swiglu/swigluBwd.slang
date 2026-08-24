import activations;
import storage;

struct PushConstants {
	uint gate_idx;
	uint up_idx;
	uint d_out_idx;
	uint d_gate_idx;
	uint d_up_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float g = oaLoad(heap[pc.gate_idx], idx);
	float dout = oaLoad(heap[pc.d_out_idx], idx);

	oaStore(heap[pc.d_gate_idx], idx, dout * oaLoad(heap[pc.up_idx], idx) * SiluGrad(g));
	oaStore(heap[pc.d_up_idx], idx, dout * Silu(g));
}
