// Combined backward for MoeCombine. PackedSlot maps packed row -> route slot;
// Inverse maps route slot -> packed row. Both outputs are fully written.
import Storage;
struct PushConstants { uint dout_idx; uint packed_idx; uint gate_idx; uint inverse_idx; uint packedSlot_idx; uint dpacked_idx; uint dgate_idx; uint T; uint K; uint D; uint R; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.R * pc.D) {
		uint packed = idx / pc.D, d = idx % pc.D;
		uint slot = heap[pc.packedSlot_idx].Load(packed * 4);
		uint token = slot / pc.K;
		float v = OaLoad(heap[pc.dout_idx], token * pc.D + d) *
			OaLoad(heap[pc.gate_idx], slot);
		OaStore(heap[pc.dpacked_idx], idx, v);
	}
	if (idx < pc.R) {
		uint slot = idx;
		uint token = slot / pc.K;
		uint packed = heap[pc.inverse_idx].Load(slot * 4);
		float sum = 0.0;
		for (uint d = 0; d < pc.D; ++d)
			sum += OaLoad(heap[pc.dout_idx], token * pc.D + d) *
				OaLoad(heap[pc.packed_idx], packed * pc.D + d);
		OaStore(heap[pc.dgate_idx], slot, sum);
	}
}
