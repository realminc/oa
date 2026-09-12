// Mechanical adaptation of OA rollout/rlRolloutReset.slang.

import storage;
import attributes;

struct PushConstants {
	uint valid_index;
	uint count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("reset")]
[domain("ml")]
[variant("categorical_time_major")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) {
		return;
	}
	store_u8(storage_buffers[push.valid_index], index, 0u);
}
