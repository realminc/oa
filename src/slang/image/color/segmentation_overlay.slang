import attributes;
import storage;

struct PushConstants {
	uint image_index, mask_index, palette_index, output_index;
	uint batch_size, channels, height, width, class_count;
	float alpha;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("segmentation_overlay")]
[domain("image")]
[variant("i32_labels_f32_palette")]
[dtype("f32")]
[layout("nchw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane_index = thread_id.z;
	if (x >= push.width || y >= push.height || plane_index >= push.batch_size * push.channels) return;
	uint batch = plane_index / push.channels, channel = plane_index % push.channels;
	uint pixel = (batch * push.height + y) * push.width + x;
	uint image_element = (plane_index * push.height + y) * push.width + x;
	float value = load_f32(storage_buffers[push.image_index], image_element);
	int label = load_i32(storage_buffers[push.mask_index], pixel);
	if (label >= 0 && uint(label) < push.class_count && channel < 3u) {
		float color = load_f32(storage_buffers[push.palette_index], uint(label) * 3u + channel);
		value = value * (1.0f - push.alpha) + color * push.alpha;
	}
	store_f32(storage_buffers[push.output_index], image_element, value);
}
