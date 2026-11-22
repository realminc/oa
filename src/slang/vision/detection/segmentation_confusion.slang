import storage;
import attributes;

struct PushConstants {
	uint predicted_index;
	uint target_index;
	uint output_index;
	uint element_count;
	uint class_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("segmentation_confusion")]
[domain("vision")]
[variant("atomic_u32")]
[dtype("i32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;
	int predicted = load_i32(storage_buffers[push.predicted_index], index);
	int target = load_i32(storage_buffers[push.target_index], index);
	if (predicted < 0 || target < 0 ||
		predicted >= int(push.class_count) || target >= int(push.class_count)) return;
	uint ignored;
	storage_buffers[push.output_index].InterlockedAdd(
		(uint(target) * push.class_count + uint(predicted)) * 4u, 1u, ignored);
}
