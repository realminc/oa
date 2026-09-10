import storage;
import attributes;

struct PushConstants {
	uint a_index, b_index, mask_index, output_index;
	uint batch_size, channels, height, width, operation, mask_channels;
	uint rect_x, rect_y, rect_width, rect_height;
	float alpha, value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("composite")]
[domain("image")]
[variant("parameterized")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint batch = plane / push.channels, channel = plane % push.channels;
	uint index = (plane * push.height + y) * push.width + x;
	float a = load_f32(storage_buffers[push.a_index], index);
	float result = a;
	if (push.operation == 0u) {
		float b = load_f32(storage_buffers[push.b_index], index);
		result = a * (1.0f - push.alpha) + b * push.alpha;
	} else if (push.operation == 1u) {
		float b = load_f32(storage_buffers[push.b_index], index);
		uint mask_channel = push.mask_channels == 1u ? 0u : channel;
		uint mask_index = ((batch * push.mask_channels + mask_channel) * push.height + y) * push.width + x;
		float mask = clamp(load_f32(storage_buffers[push.mask_index], mask_index), 0.0f, 1.0f);
		result = a * (1.0f - mask) + b * mask;
	} else {
		bool inside = x >= push.rect_x && x - push.rect_x < push.rect_width &&
			y >= push.rect_y && y - push.rect_y < push.rect_height;
		result = inside ? push.value : a;
	}
	store_f32(storage_buffers[push.output_index], index, result);
}
