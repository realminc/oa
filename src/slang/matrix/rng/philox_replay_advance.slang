import attributes;

struct PushConstants {
	uint state_index;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("philox_replay_advance")]
[domain("matrix")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	if (dispatch_thread_id.x != 0u) {
		return;
	}
	uint previous;
	storage_buffers[push.state_index].InterlockedAdd(0u, 1u, previous);
}
