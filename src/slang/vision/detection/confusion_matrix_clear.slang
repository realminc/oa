import storage;
import attributes;

struct PushConstants {
	uint output_index;
	uint element_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("confusion_matrix_clear")]
[domain("vision")]
[variant("replay_safe_zero")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;
	store_u32(storage_buffers[push.output_index], index, 0u);
}
