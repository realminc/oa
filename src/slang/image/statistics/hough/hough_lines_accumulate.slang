import attributes;
import storage;

struct PushConstants {
	uint input_index, acc_index;
	uint batch_size, channels, height, width;
	float edge_threshold, rho_step, theta_step;
	uint num_theta, num_rho;
	float rho_span;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("hough_lines_accumulate")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint index = plane * push.height * push.width + y * push.width + x;
	float value = load_f32(storage_buffers[push.input_index], index);
	if (value <= push.edge_threshold) return;
	uint bins = push.num_theta * push.num_rho;
	uint acc_base = plane * (bins + 1u);
	float radians_step = push.theta_step * 3.14159265 / 180.0;
	for (uint ti = 0; ti < push.num_theta; ++ti) {
		float theta = float(ti) * radians_step;
		float rho = float(x) * cos(theta) + float(y) * sin(theta);
		float bin = (rho + push.rho_span) / push.rho_step;
		uint ri = uint(floor(bin + 0.5));
		if (ri >= push.num_rho) continue;
		storage_buffers[push.acc_index].InterlockedAdd((acc_base + ti * push.num_rho + ri) * 4u, 1u);
	}
}
