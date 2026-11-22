import storage;
import attributes;

struct PushConstants {
	uint predicted_index;
	uint target_index;
	uint output_index;
	uint element_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("binary_mask_counts")]
[domain("vision")]
[variant("atomic_u32")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;
	bool predicted = load_u8(storage_buffers[push.predicted_index], index) != 0u;
	bool target = load_u8(storage_buffers[push.target_index], index) != 0u;
	uint bucket = predicted ? (target ? 0u : 1u) : (target ? 2u : 3u);
	uint ignored;
	storage_buffers[push.output_index].InterlockedAdd(bucket * 4u, 1u, ignored);
}
