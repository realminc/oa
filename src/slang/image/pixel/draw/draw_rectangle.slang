import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, element_count, height, width;
	float x, y, rect_width, rect_height, color, thickness;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("draw_rectangle")]
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
	float right = push.x + push.rect_width;
	float bottom = push.y + push.rect_height;
	bool inside = px >= push.x && px < right && py >= push.y && py < bottom;
	float edge = min(min(px - push.x, right - px), min(py - push.y, bottom - py));
	bool covered = inside && (push.thickness < 0.0f || edge <= push.thickness);
	float value = load_f32(storage_buffers[push.input_index], index);
	store_f32(
		storage_buffers[push.output_index],
		index,
		covered ? push.color : value);
}
