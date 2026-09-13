// Inverse rotary transform for MIMO Q/K gradients plus cumulative-angle
// adjoint. One invocation owns (batch, head), avoiding atomics.

import storage;
import attributes;

static const uint MAXA = 64u;

struct PushConstants {
	uint dq_idx, dk_idx, c_idx, b_idx, dt_idx, angle_idx;
	uint cbias_idx, bbias_idx, theta_idx;
	uint dc_idx, db_idx, dangle_head_idx, dc_bias_batch_idx, db_bias_batch_idx;
	uint ddt_idx;
	uint b, l, h, g, n, a, r;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_mimo_backward_rotate")]
[domain("ml")]
[variant("rotate")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint g = h / (push.h / push.g);
	uint biasBatch = bh * push.r * push.n;
	for (uint i = 0u; i < push.r * push.n; ++i) {
		store_f32(storage_buffers[push.dc_bias_batch_idx], biasBatch + i, 0.0f);
		store_f32(storage_buffers[push.db_bias_batch_idx], biasBatch + i, 0.0f);
	}
	float running[MAXA];
	for (uint a = 0u; a < push.a; ++a) running[a] = 0.0f;

	for (int ti = int(push.l) - 1; ti >= 0; --ti) {
		uint t = uint(ti);
		uint scalar = (b * push.l + t) * push.h + h;
		float dTheta[MAXA];
		for (uint a = 0u; a < push.a; ++a) dTheta[a] = 0.0f;
		for (uint r = 0u; r < push.r; ++r) {
			for (uint n = 0u; n < push.n; ++n) {
				uint qk = (scalar * push.r + r) * push.n + n;
				float dRawQ = load_f32(storage_buffers[push.dq_idx], qk);
				float dRawK = load_f32(storage_buffers[push.dk_idx], qk);
				uint raw = (((b * push.l + t) * (push.r * push.g) + (r * push.g + g)) * push.n) + n;
				uint bias = (h * push.r + r) * push.n + n;
				if (n < 2u * push.a) {
					uint a = n / 2u;
					uint n0 = 2u * a, n1 = n0 + 1u;
					uint qk0 = qk - (n - n0);
					float dq0 = load_f32(storage_buffers[push.dq_idx], qk0);
					float dq1 = load_f32(storage_buffers[push.dq_idx], qk0 + 1u);
					float dk0 = load_f32(storage_buffers[push.dk_idx], qk0);
					float dk1 = load_f32(storage_buffers[push.dk_idx], qk0 + 1u);
					float theta = load_f32(storage_buffers[push.theta_idx], (bh * push.l + t) * push.a + a);
					float cs = cos(theta), sn = sin(theta);
					if (n == n0) {
						dRawQ = dq0 * cs + dq1 * sn;
						dRawK = dk0 * cs + dk1 * sn;
						float q0 = load_f32(storage_buffers[push.c_idx], raw) + load_f32(storage_buffers[push.cbias_idx], bias);
						float q1 = load_f32(storage_buffers[push.c_idx], raw + 1u) + load_f32(storage_buffers[push.cbias_idx], bias + 1u);
						float k0 = load_f32(storage_buffers[push.b_idx], raw) + load_f32(storage_buffers[push.bbias_idx], bias);
						float k1 = load_f32(storage_buffers[push.b_idx], raw + 1u) + load_f32(storage_buffers[push.bbias_idx], bias + 1u);
						dTheta[a] += dq0 * (-q0 * sn - q1 * cs)
							+ dq1 * (q0 * cs - q1 * sn)
							+ dk0 * (-k0 * sn - k1 * cs)
							+ dk1 * (k0 * cs - k1 * sn);
					} else {
						dRawQ = -dq0 * sn + dq1 * cs;
						dRawK = -dk0 * sn + dk1 * cs;
					}
				}
				uint headGrad = ((scalar * push.r + r) * push.n) + n;
				store_f32(storage_buffers[push.dc_idx], headGrad, dRawQ);
				store_f32(storage_buffers[push.db_idx], headGrad, dRawK);
				uint partial = biasBatch + r * push.n + n;
				store_f32(storage_buffers[push.dc_bias_batch_idx], partial,
					load_f32(storage_buffers[push.dc_bias_batch_idx], partial) + dRawQ);
				store_f32(storage_buffers[push.db_bias_batch_idx], partial,
					load_f32(storage_buffers[push.db_bias_batch_idx], partial) + dRawK);
			}
		}
		float dtAngle = 0.0f;
		uint angleBase = (b * push.l + t) * push.a;
		for (uint a = 0u; a < push.a; ++a) {
			running[a] += dTheta[a];
			store_f32(storage_buffers[push.dangle_head_idx], (bh * push.l + t) * push.a + a,
				running[a] * load_f32(storage_buffers[push.dt_idx], scalar));
			dtAngle += running[a] * load_f32(storage_buffers[push.angle_idx], angleBase + a);
		}
		store_f32(storage_buffers[push.ddt_idx], scalar,
			load_f32(storage_buffers[push.ddt_idx], scalar) + dtAngle);
	}
}
