// Inverse rotary transform for MIMO Q/K gradients plus cumulative-angle
// adjoint. One invocation owns (batch, head), avoiding atomics.

import storage;

static const uint MAXA = 64u;

struct PushConstants {
	uint dq_idx, dk_idx, c_idx, b_idx, dt_idx, angle_idx;
	uint cbias_idx, bbias_idx, theta_idx;
	uint dc_idx, db_idx, dangle_head_idx, dc_bias_batch_idx, db_bias_batch_idx;
	uint ddt_idx;
	uint B, L, H, G, N, A, R;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint g = h / (pc.H / pc.G);
	uint biasBatch = bh * pc.R * pc.N;
	for (uint i = 0u; i < pc.R * pc.N; ++i) {
		oaStoreF32(heap[pc.dc_bias_batch_idx], biasBatch + i, 0.0f);
		oaStoreF32(heap[pc.db_bias_batch_idx], biasBatch + i, 0.0f);
	}
	float running[MAXA];
	for (uint a = 0u; a < pc.A; ++a) running[a] = 0.0f;

	for (int ti = int(pc.L) - 1; ti >= 0; --ti) {
		uint t = uint(ti);
		uint scalar = (b * pc.L + t) * pc.H + h;
		float dTheta[MAXA];
		for (uint a = 0u; a < pc.A; ++a) dTheta[a] = 0.0f;
		for (uint r = 0u; r < pc.R; ++r) {
			for (uint n = 0u; n < pc.N; ++n) {
				uint qk = (scalar * pc.R + r) * pc.N + n;
				float dRawQ = oaLoadF32(heap[pc.dq_idx], qk);
				float dRawK = oaLoadF32(heap[pc.dk_idx], qk);
				uint raw = (((b * pc.L + t) * (pc.R * pc.G) + (r * pc.G + g)) * pc.N) + n;
				uint bias = (h * pc.R + r) * pc.N + n;
				if (n < 2u * pc.A) {
					uint a = n / 2u;
					uint n0 = 2u * a, n1 = n0 + 1u;
					uint qk0 = qk - (n - n0);
					float dq0 = oaLoadF32(heap[pc.dq_idx], qk0);
					float dq1 = oaLoadF32(heap[pc.dq_idx], qk0 + 1u);
					float dk0 = oaLoadF32(heap[pc.dk_idx], qk0);
					float dk1 = oaLoadF32(heap[pc.dk_idx], qk0 + 1u);
					float theta = oaLoadF32(heap[pc.theta_idx], (bh * pc.L + t) * pc.A + a);
					float cs = cos(theta), sn = sin(theta);
					if (n == n0) {
						dRawQ = dq0 * cs + dq1 * sn;
						dRawK = dk0 * cs + dk1 * sn;
						float q0 = oaLoadF32(heap[pc.c_idx], raw) + oaLoadF32(heap[pc.cbias_idx], bias);
						float q1 = oaLoadF32(heap[pc.c_idx], raw + 1u) + oaLoadF32(heap[pc.cbias_idx], bias + 1u);
						float k0 = oaLoadF32(heap[pc.b_idx], raw) + oaLoadF32(heap[pc.bbias_idx], bias);
						float k1 = oaLoadF32(heap[pc.b_idx], raw + 1u) + oaLoadF32(heap[pc.bbias_idx], bias + 1u);
						dTheta[a] += dq0 * (-q0 * sn - q1 * cs)
							+ dq1 * (q0 * cs - q1 * sn)
							+ dk0 * (-k0 * sn - k1 * cs)
							+ dk1 * (k0 * cs - k1 * sn);
					} else {
						dRawQ = -dq0 * sn + dq1 * cs;
						dRawK = -dk0 * sn + dk1 * cs;
					}
				}
				uint headGrad = ((scalar * pc.R + r) * pc.N) + n;
				oaStoreF32(heap[pc.dc_idx], headGrad, dRawQ);
				oaStoreF32(heap[pc.db_idx], headGrad, dRawK);
				uint partial = biasBatch + r * pc.N + n;
				oaStoreF32(heap[pc.dc_bias_batch_idx], partial,
					oaLoadF32(heap[pc.dc_bias_batch_idx], partial) + dRawQ);
				oaStoreF32(heap[pc.db_bias_batch_idx], partial,
					oaLoadF32(heap[pc.db_bias_batch_idx], partial) + dRawK);
			}
		}
		float dtAngle = 0.0f;
		uint angleBase = (b * pc.L + t) * pc.A;
		for (uint a = 0u; a < pc.A; ++a) {
			running[a] += dTheta[a];
			oaStoreF32(heap[pc.dangle_head_idx], (bh * pc.L + t) * pc.A + a,
				running[a] * oaLoadF32(heap[pc.dt_idx], scalar));
			dtAngle += running[a] * oaLoadF32(heap[pc.angle_idx], angleBase + a);
		}
		oaStoreF32(heap[pc.ddt_idx], scalar,
			oaLoadF32(heap[pc.ddt_idx], scalar) + dtAngle);
	}
}
