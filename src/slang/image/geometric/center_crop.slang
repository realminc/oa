import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, input_height, input_width;
	uint crop_x, crop_y, output_height, output_width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("center_crop")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.output_width || y >= push.output_height ||
		plane >= push.batch_size * push.channels) return;
	uint source = (plane * push.input_height + push.crop_y + y) * push.input_width + push.crop_x + x;
	uint target = (plane * push.output_height + y) * push.output_width + x;
	store_f32(storage_buffers[push.output_index], target,
		load_f32(storage_buffers[push.input_index], source));
}
