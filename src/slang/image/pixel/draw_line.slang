import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, element_count, height, width;
	float x1, y1, x2, y2, color, thickness;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("draw_line")]
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
	float dx = push.x2 - push.x1;
	float dy = push.y2 - push.y1;
	float segment = dx * dx + dy * dy;
	float t = 0.0f;
	if (segment > 0.0f) {
		t = clamp(((px - push.x1) * dx + (py - push.y1) * dy) / segment, 0.0f, 1.0f);
	}
	float ox = px - (push.x1 + t * dx);
	float oy = py - (push.y1 + t * dy);
	float distance = sqrt(ox * ox + oy * oy);
	float value = load_f32(storage_buffers[push.input_index], index);
	store_f32(
		storage_buffers[push.output_index],
		index,
		distance <= push.thickness * 0.5f ? push.color : value);
}
