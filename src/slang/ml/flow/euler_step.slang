// OARS single-pass provider for OA's composite FnFlow::eulerStep contract.

import storage;
import attributes;

struct PushConstants {
	uint state_index;
	uint velocity_index;
	uint output_index;
	uint element_count;
	float delta_time;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("euler_step")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	float state = load_f32(storage_buffers[push.state_index], index);
	float velocity = load_f32(storage_buffers[push.velocity_index], index);
	store_f32(storage_buffers[push.output_index], index, state + velocity * push.delta_time);
}
