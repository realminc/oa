import attributes;
import storage;

struct PushConstants {
	uint input_index, labels_index;
	uint batch_size, channels, height, width;
	float threshold;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("connected_components_init")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint index = plane * push.height * push.width + y * push.width + x;
	float value = load_f32(storage_buffers[push.input_index], index);
	store_u32(
		storage_buffers[push.labels_index],
		index,
		value > push.threshold ? index : 0xFFFFFFFFu);
}
