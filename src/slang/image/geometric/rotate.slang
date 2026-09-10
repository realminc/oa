import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, input_height, input_width, output_height, output_width, degrees;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("rotate")]
[domain("image")]
[variant("clockwise_quarter_turns")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.output_width || y >= push.output_height ||
		plane >= push.batch_size * push.channels) return;
	uint source_x = x, source_y = y;
	if (push.degrees == 90u) {
		source_x = y;
		source_y = push.input_height - 1u - x;
	} else if (push.degrees == 180u) {
		source_x = push.input_width - 1u - x;
		source_y = push.input_height - 1u - y;
	} else if (push.degrees == 270u) {
		source_x = push.input_width - 1u - y;
		source_y = x;
	}
	uint source = (plane * push.input_height + source_y) * push.input_width + source_x;
	uint target = (plane * push.output_height + y) * push.output_width + x;
	store_f32(storage_buffers[push.output_index], target,
		load_f32(storage_buffers[push.input_index], source));
}
