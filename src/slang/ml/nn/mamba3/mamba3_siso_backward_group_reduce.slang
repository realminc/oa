// Deterministic Mamba-3 Q/K gradient reduction from heads back to groups.
//
// The forward scan maps each head to one shared Q/K group.  Backward computes
// independent per-head contributions, then this kernel restores the grouped
// input contract without atomics or a generic broadcast-autograd subgraph.

import storage;
import attributes;

static const uint THREADS = 256u;

struct PushConstants {
	uint dc_head_idx;
	uint db_head_idx;
	uint dc_group_idx;
	uint db_group_idx;
	uint b;
	uint l;
	uint h;
	uint g;
	uint n;
	uint count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_group_reduce")]
[domain("ml")]
[variant("group_reduce")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= push.count) return;
	uint n = index % push.n;
	uint tokenGroup = index / push.n;
	uint g = tokenGroup % push.g;
	uint token = tokenGroup / push.g;
	uint headsPerGroup = push.h / push.g;
	uint firstHead = g * headsPerGroup;
	float dc = 0.0f;
	float db = 0.0f;
	for (uint localHead = 0u; localHead < headsPerGroup; ++localHead) {
		uint h = firstHead + localHead;
		uint headIndex = (token * push.h + h) * push.n + n;
		dc += load_f32(storage_buffers[push.dc_head_idx], headIndex);
		db += load_f32(storage_buffers[push.db_head_idx], headIndex);
	}
	store_f32(storage_buffers[push.dc_group_idx], index, dc);
	store_f32(storage_buffers[push.db_group_idx], index, db);
}





