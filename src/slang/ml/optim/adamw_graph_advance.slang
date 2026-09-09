// Replay-safe AdamW step-state advance.

import storage;
import attributes;

struct PushConstants {
	uint state_index;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adamw_graph_advance")]
[domain("ml")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	if (dispatch_thread_id.x != 0u) {
		return;
	}
	uint step = storage_buffers[push.state_index].Load(0u);
	storage_buffers[push.state_index].Store(0u, step + 1u);
}
