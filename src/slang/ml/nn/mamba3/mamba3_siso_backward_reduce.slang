// Deterministic reduction for the short Mamba-3 adjoint. The recurrent stages
// write one partial per (batch, head), avoiding contended global floating-point
// atomics for the shared angle, bias, and skip-parameter gradients.

import storage;
import attributes;

struct PushConstants {
	uint dangle_head_idx;
	uint dc_batch_idx;
	uint db_batch_idx;
	uint dd_batch_idx;
	uint dangle_idx;
	uint dc_idx;
	uint db_idx;
	uint dd_idx;
	uint b;
	uint l;
	uint h;
	uint n;
	uint a;
	uint count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_reduce")]
[domain("ml")]
[variant("short_reduce")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= push.count) return;
	uint angleCount = push.b * push.l * push.a;
	uint hn = push.h * push.n;
	if (index < angleCount) {
		uint k = index % push.a;
		uint token = index / push.a;
		uint t = token % push.l;
		uint b = token / push.l;
		float value = 0.0f;
		for (uint h = 0u; h < push.h; ++h) {
			value += load_f32(storage_buffers[push.dangle_head_idx],
				((b * push.h + h) * push.l + t) * push.a + k);
		}
		store_f32(storage_buffers[push.dangle_idx], index, value);
	} else if (index < angleCount + hn) {
		uint local = index - angleCount;
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			value += load_f32(storage_buffers[push.dc_batch_idx], b * hn + local);
		}
		store_f32(storage_buffers[push.dc_idx], local, value);
	} else if (index < angleCount + 2u * hn) {
		uint local = index - angleCount - hn;
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			value += load_f32(storage_buffers[push.db_batch_idx], b * hn + local);
		}
		store_f32(storage_buffers[push.db_idx], local, value);
	} else {
		uint h = index - angleCount - 2u * hn;
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			value += load_f32(storage_buffers[push.dd_batch_idx], b * push.h + h);
		}
		store_f32(storage_buffers[push.dd_idx], h, value);
	}
}





