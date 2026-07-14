// Direct token-major weighted combine from expert-major packed rows.
// Inverse maps route slot (t*K+k) -> packed row, so every output element has a
// single writer and no atomics/zero-fill pass is required.
import Storage;
struct PushConstants { uint packed_idx; uint gate_idx; uint inverse_idx; uint out_idx; uint T; uint K; uint D; uint R; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.T * pc.D) return;
	uint t = idx / pc.D, d = idx % pc.D;
	float sum = 0.0;
	for (uint k = 0; k < pc.K; ++k) {
		uint slot = t * pc.K + k;
		uint packed = heap[pc.inverse_idx].Load(slot * 4);
		sum += OaLoad(heap[pc.packed_idx], packed * pc.D + d) *
			OaLoad(heap[pc.gate_idx], slot);
	}
	OaStore(heap[pc.out_idx], idx, sum);
}
