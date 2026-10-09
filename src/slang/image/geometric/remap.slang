import storage;
import attributes;

struct PushConstants {
	uint input_index, map_index, output_index;
	uint batch_size, channels, input_height, input_width, output_height, output_width;
	uint interpolation, border_mode, map_batch;
	float border_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int border_coordinate(int coordinate, uint extent) {
	if (coordinate >= 0 && coordinate < int(extent)) return coordinate;
	if (push.border_mode == 0u) return -1;
	if (push.border_mode == 1u) return clamp(coordinate, 0, int(extent) - 1);
	if (extent == 1u) return 0;
	int period = push.border_mode == 2u ? int(2u * extent) : int(2u * (extent - 1u));
	if (push.border_mode == 4u) period = int(extent);
	int folded = coordinate % period;
	if (folded < 0) folded += period;
	if (push.border_mode == 4u || folded < int(extent)) return folded;
	return push.border_mode == 2u ? period - 1 - folded : period - folded;
}

float sample_one(uint plane, int y, int x) {
	int source_x = border_coordinate(x, push.input_width);
	int source_y = border_coordinate(y, push.input_height);
	if (source_x < 0 || source_y < 0) return push.border_value;
	uint source = (plane * push.input_height + uint(source_y)) * push.input_width + uint(source_x);
	return load_f32(storage_buffers[push.input_index], source);
}

float sample_image(uint plane, float y, float x) {
	if (push.interpolation == 0u)
		return sample_one(plane, int(floor(y + 0.5f)), int(floor(x + 0.5f)));
	int y0 = int(floor(y)), x0 = int(floor(x));
	float fy = y - float(y0), fx = x - float(x0);
	return lerp(lerp(sample_one(plane, y0, x0), sample_one(plane, y0, x0 + 1), fx),
		lerp(sample_one(plane, y0 + 1, x0), sample_one(plane, y0 + 1, x0 + 1), fx), fy);
}

[kernel_name("remap")]
[domain("image")]
[variant("absolute_coordinates")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.output_width || y >= push.output_height ||
		plane >= push.batch_size * push.channels) return;
	uint batch = plane / push.channels;
	uint map_plane = push.output_height * push.output_width;
	uint map_base = (push.map_batch == 1u ? 0u : batch) * 2u * map_plane + y * push.output_width + x;
	float source_x = load_f32(storage_buffers[push.map_index], map_base);
	float source_y = load_f32(storage_buffers[push.map_index], map_base + map_plane);
	uint target = (plane * push.output_height + y) * push.output_width + x;
	store_f32(storage_buffers[push.output_index], target, sample_image(plane, source_y, source_x));
}
