import storage;
import attributes;

struct PushConstants {
	uint boxes_a_index;
	uint boxes_b_index;
	uint output_index;
	uint rows_a;
	uint rows_b;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

float pair_iou(uint row_a, uint row_b) {
	uint a = row_a * 4u;
	uint b = row_b * 4u;
	float a_center_x = load_f32(storage_buffers[push.boxes_a_index], a);
	float a_center_y = load_f32(storage_buffers[push.boxes_a_index], a + 1u);
	float a_width = load_f32(storage_buffers[push.boxes_a_index], a + 2u);
	float a_height = load_f32(storage_buffers[push.boxes_a_index], a + 3u);
	float b_center_x = load_f32(storage_buffers[push.boxes_b_index], b);
	float b_center_y = load_f32(storage_buffers[push.boxes_b_index], b + 1u);
	float b_width = load_f32(storage_buffers[push.boxes_b_index], b + 2u);
	float b_height = load_f32(storage_buffers[push.boxes_b_index], b + 3u);
	if (!isfinite(a_center_x) || !isfinite(a_center_y) || !isfinite(a_width) ||
		!isfinite(a_height) || !isfinite(b_center_x) || !isfinite(b_center_y) ||
		!isfinite(b_width) || !isfinite(b_height)) return 0.0f;

	a_width = max(a_width, 0.0f);
	a_height = max(a_height, 0.0f);
	b_width = max(b_width, 0.0f);
	b_height = max(b_height, 0.0f);
	float a_x0 = a_center_x - a_width * 0.5f;
	float a_y0 = a_center_y - a_height * 0.5f;
	float a_x1 = a_center_x + a_width * 0.5f;
	float a_y1 = a_center_y + a_height * 0.5f;
	float b_x0 = b_center_x - b_width * 0.5f;
	float b_y0 = b_center_y - b_height * 0.5f;
	float b_x1 = b_center_x + b_width * 0.5f;
	float b_y1 = b_center_y + b_height * 0.5f;
	float intersection_width = max(min(a_x1, b_x1) - max(a_x0, b_x0), 0.0f);
	float intersection_height = max(min(a_y1, b_y1) - max(a_y0, b_y0), 0.0f);
	float intersection = intersection_width * intersection_height;
	float union_area = a_width * a_height + b_width * b_height - intersection;
	return isfinite(union_area) && union_area > 0.0f ? intersection / union_area : 0.0f;
}

[kernel_name("box_iou")]
[domain("vision")]
[variant("pairwise_center_xywh")]
[dtype("f32")]
[layout("row_major_n4")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint pair = thread_id.x;
	uint pair_count = push.rows_a * push.rows_b;
	if (pair >= pair_count) return;
	uint row_a = pair / push.rows_b;
	uint row_b = pair - row_a * push.rows_b;
	store_f32(storage_buffers[push.output_index], pair, pair_iou(row_a, row_b));
}
