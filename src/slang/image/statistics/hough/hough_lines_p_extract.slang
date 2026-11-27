import attributes;
import storage;

struct PushConstants {
	uint acc_index, lines_index;
	uint num_theta, num_rho, plane_count;
	uint threshold, max_lines;
	float rho_step, theta_step, rho_span;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("hough_lines_p_extract")]
[domain("image")]
[variant("replay_safe_zero")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint cell = thread_id.x;
	uint bins = push.num_theta * push.num_rho;
	uint total = push.plane_count * bins;
	if (cell >= total) return;
	uint plane = cell / bins;
	uint ti = (cell % bins) / push.num_rho;
	uint ri = (cell % bins) % push.num_rho;
	RWByteAddressBuffer acc = storage_buffers[push.acc_index];
	uint acc_base = plane * (bins + 1u);
	uint votes = load_u32(acc, acc_base + ti * push.num_rho + ri);
	if (votes < push.threshold || votes == 0u) return;
	for (int dti = -1; dti <= 1; ++dti) {
		for (int dri = -1; dri <= 1; ++dri) {
			if (dti == 0 && dri == 0) continue;
			int nti = int(ti) + dti;
			int nri = int(ri) + dri;
			if (nti < 0 || nri < 0 || nti >= int(push.num_theta) || nri >= int(push.num_rho)) continue;
			uint neighbor = acc_base + uint(nti) * push.num_rho + uint(nri);
			uint other = load_u32(acc, neighbor);
			uint neighbor_cell = plane * bins + uint(nti) * push.num_rho + uint(nri);
			if (other > votes || (other == votes && neighbor_cell < cell)) return;
		}
	}
	uint slot;
	acc.InterlockedAdd((acc_base + bins) * 4u, 1u, slot);
	if (slot >= push.max_lines) return;
	uint out_base = plane * push.max_lines * 3u + slot * 3u;
	RWByteAddressBuffer lines = storage_buffers[push.lines_index];
	store_f32(lines, out_base, float(ri) * push.rho_step - push.rho_span);
	store_f32(lines, out_base + 1u, float(ti) * push.theta_step);
	store_f32(lines, out_base + 2u, float(votes));
}
