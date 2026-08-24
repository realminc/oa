// Deterministic reduction for the short Mamba-3 adjoint. The recurrent stages
// write one partial per (batch, head), avoiding contended global floating-point
// atomics for the shared angle, bias, and skip-parameter gradients.

import storage;

struct PushConstants {
	uint dangle_head_idx;
	uint dc_batch_idx;
	uint db_batch_idx;
	uint dd_batch_idx;
	uint dangle_idx;
	uint dc_idx;
	uint db_idx;
	uint dd_idx;
	uint B;
	uint L;
	uint H;
	uint N;
	uint A;
	uint count;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= pc.count) return;
	uint angleCount = pc.B * pc.L * pc.A;
	uint hn = pc.H * pc.N;
	if (index < angleCount) {
		uint k = index % pc.A;
		uint token = index / pc.A;
		uint t = token % pc.L;
		uint b = token / pc.L;
		float value = 0.0f;
		for (uint h = 0u; h < pc.H; ++h) {
			value += oaLoadF32(heap[pc.dangle_head_idx],
				((b * pc.H + h) * pc.L + t) * pc.A + k);
		}
		oaStoreF32(heap[pc.dangle_idx], index, value);
	} else if (index < angleCount + hn) {
		uint local = index - angleCount;
		float value = 0.0f;
		for (uint b = 0u; b < pc.B; ++b) {
			value += oaLoadF32(heap[pc.dc_batch_idx], b * hn + local);
		}
		oaStoreF32(heap[pc.dc_idx], local, value);
	} else if (index < angleCount + 2u * hn) {
		uint local = index - angleCount - hn;
		float value = 0.0f;
		for (uint b = 0u; b < pc.B; ++b) {
			value += oaLoadF32(heap[pc.db_batch_idx], b * hn + local);
		}
		oaStoreF32(heap[pc.db_idx], local, value);
	} else {
		uint h = index - angleCount - 2u * hn;
		float value = 0.0f;
		for (uint b = 0u; b < pc.B; ++b) {
			value += oaLoadF32(heap[pc.dd_batch_idx], b * pc.H + h);
		}
		oaStoreF32(heap[pc.dd_idx], h, value);
	}
}
