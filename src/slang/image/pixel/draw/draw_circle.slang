import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, element_count, height, width;
	float cx, cy, radius, color, thickness;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("draw_circle")]
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
	float dx = px - push.cx;
	float dy = py - push.cy;
	float distance = sqrt(dx * dx + dy * dy);
	bool covered;
	if (push.thickness < 0.0f) {
		covered = distance <= push.radius;
	} else {
		covered = abs(distance - push.radius) <= push.thickness * 0.5f;
	}
	float value = load_f32(storage_buffers[push.input_index], index);
	store_f32(
		storage_buffers[push.output_index],
		index,
		covered ? push.color : value);
}
