import storage;
import attributes;

struct PushConstants {
	uint input_index, points_index, output_index;
	uint batch_size, channels, element_count, height, width;
	uint point_count;
	float color;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("fill_polygon")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;
	uint plane_size = push.height * push.width;
	uint within = index % plane_size;
	float px = float(within % push.width) + 0.5f;
	float py = float(within / push.width) + 0.5f;
	RWByteAddressBuffer points_buffer = storage_buffers[push.points_index];
	bool inside = false;
	uint previous = push.point_count - 1u;
	for (uint i = 0u; i < push.point_count; ++i) {
		float xi = load_f32(points_buffer, i * 2u);
		float yi = load_f32(points_buffer, i * 2u + 1u);
		float xj = load_f32(points_buffer, previous * 2u);
		float yj = load_f32(points_buffer, previous * 2u + 1u);
		if ((yi > py) != (yj > py) &&
			px < xj + (xi - xj) * (py - yj) / (yi - yj)) {
			inside = !inside;
		}
		previous = i;
	}
	float value = load_f32(storage_buffers[push.input_index], index);
	store_f32(
		storage_buffers[push.output_index],
		index,
		inside ? push.color : value);
}
