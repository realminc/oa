import attributes;
import storage;

struct PushConstants {
	uint cost_index, output_index;
	uint batch_size, channels, height, width;
	uint num_disparities, block_radius, uniqueness;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// StereoBM winner-take-all pass. For each pixel it scans the SAD
// volume over the full disparity range, keeps the argmin, applies
// the uniqueness test (reject when the runner-up is within
// `uniqueness` percent of the best cost), and refines the winning
// disparity with a symmetric parabola on the neighbouring costs.
// Pixels whose matching window cannot stay in-bounds for the full
// disparity range, and uniqueness losers, are written as -1.
[kernel_name("stereo_wta")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	uint plane_count = push.batch_size * push.channels;
	if (x >= push.width || y >= push.height || plane >= plane_count) return;
	RWByteAddressBuffer cost_buffer = storage_buffers[push.cost_index];
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	uint plane_size = push.height * push.width;
	uint out_index = plane * plane_size + y * push.width + x;
	uint radius = push.block_radius;
	// Valid region: the full matching window must fit in-bounds at
	// every candidate disparity, matching StereoBM's border policy.
	bool in_bounds =
		x >= radius + push.num_disparities - 1u &&
		x + radius < push.width &&
		y >= radius &&
		y + radius < push.height;
	if (!in_bounds) {
		store_f32(output_buffer, out_index, -1.0f);
		return;
	}
	uint cost_base = plane * push.num_disparities * plane_size + y * push.width + x;
	float best = load_f32(cost_buffer, cost_base);
	uint best_d = 0u;
	float second = 1.0e30f;
	for (uint d = 1u; d < push.num_disparities; ++d) {
		float value = load_f32(cost_buffer, cost_base + d * plane_size);
		if (value < best) {
			second = best;
			best = value;
			best_d = d;
		} else if (value < second) {
			second = value;
		}
	}
	if (push.uniqueness > 0u &&
		second <= best * (1.0f + float(push.uniqueness) * 0.01f)) {
		store_f32(output_buffer, out_index, -1.0f);
		return;
	}
	float disparity = float(best_d);
	if (best_d > 0u && best_d + 1u < push.num_disparities) {
		float prev_cost = load_f32(cost_buffer, cost_base + (best_d - 1u) * plane_size);
		float next_cost = load_f32(cost_buffer, cost_base + (best_d + 1u) * plane_size);
		float denominator = prev_cost + next_cost - 2.0f * best;
		if (denominator > 0.0f) {
			disparity += clamp(0.5f * (prev_cost - next_cost) / denominator, -1.0f, 1.0f);
		}
	}
	store_f32(output_buffer, out_index, disparity);
}
