import attributes;
import storage;

struct PushConstants {
	uint keypoints_index, descriptors_index, counts_index;
	uint batch_size, channels, max_features;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Zero the keypoint, descriptor, and claimed-slot workspaces before the
// describe dispatches claim rows through the shared atomic counter.
[kernel_name("orb_clear")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	uint descriptor_elements = plane_count * push.max_features * 8;
	if (index >= descriptor_elements) return;
	store_u32(storage_buffers[push.descriptors_index], index, 0u);
	uint keypoint_elements = plane_count * push.max_features * 6;
	if (index < keypoint_elements)
		store_f32(storage_buffers[push.keypoints_index], index, 0.0);
	if (index < plane_count)
		store_u32(storage_buffers[push.counts_index], index, 0u);
}
