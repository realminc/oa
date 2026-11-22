import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, input_height, input_width, output_height, output_width;
	uint pad_top, pad_left, border_mode;
	float border_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int border_coordinate(int coordinate, uint extent, uint mode) {
	if (coordinate >= 0 && coordinate < int(extent)) return coordinate;
	if (mode == 0u) return -1;
	if (mode == 1u) return clamp(coordinate, 0, int(extent) - 1);
	if (extent == 1u) return 0;
	int period = mode == 2u ? int(2u * extent) : int(2u * (extent - 1u));
	if (mode == 4u) period = int(extent);
	int folded = coordinate % period;
	if (folded < 0) folded += period;
	if (mode == 4u || folded < int(extent)) return folded;
	return mode == 2u ? period - 1 - folded : period - folded;
}

[kernel_name("pad")]
[domain("image")]
[variant("shared_border_modes")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.output_width || y >= push.output_height ||
		plane >= push.batch_size * push.channels) return;
	int source_x = border_coordinate(int(x) - int(push.pad_left), push.input_width, push.border_mode);
	int source_y = border_coordinate(int(y) - int(push.pad_top), push.input_height, push.border_mode);
	float value = push.border_value;
	if (source_x >= 0 && source_y >= 0) {
		uint source = (plane * push.input_height + uint(source_y)) * push.input_width + uint(source_x);
		value = load_f32(storage_buffers[push.input_index], source);
	}
	uint target = (plane * push.output_height + y) * push.output_width + x;
	store_f32(storage_buffers[push.output_index], target, value);
}
