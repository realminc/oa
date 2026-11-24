// AdamW-owned replay-safe step-state advance gated by a device finite flag.

import storage;
import attributes;

struct PushConstants {
	uint state_index;
	uint flag_index;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adamw_gated_graph_advance")]
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
	float flag = asfloat(storage_buffers[push.flag_index].Load(0u));
	if (flag <= 0.5f) {
		return;
	}
	uint previous;
	storage_buffers[push.state_index].InterlockedAdd(0u, 1u, previous);
}
