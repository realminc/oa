// RmsNormGated backward (norm_before_gate = 1).
//
// Forward (per row, cols i; r = 1/sqrt(mean(x^2)+eps)):
//   m[i]   = x[i]*r*w[i] + bias[i]
//   g[i]   = silu(z[i]) = z[i]*sigmoid(z[i])
//   out[i] = m[i]*g[i]
//
// Grads (dm[i] = dy[i]*g[i]):
//   c        = sum_i dm[i]*w[i]*x[i]
//   dx[i]    = r*(dm[i]*w[i] - x[i]*c*r^2/n)              (same shape as plain RMSNorm)
//   dz[i]    = dy[i]*m[i]*silu'(z[i])
//   dw[i]   += dm[i]*x[i]*r            (host-reduced over rows)
//   dbias[i]+= dm[i]                   (host-reduced over rows, if has_bias)

import Reductions;
import Storage;
import Activations;  // Silu, Sigmoid — shared module (was inline; byte-identical)

struct PushConstants {
	uint x_idx;
	uint w_idx;
	uint bias_idx;
	uint z_idx;
	uint dy_idx;
	uint dx_idx;
	uint dz_idx;
	uint dw_idx;     // [rows, cols] contributions
	uint dbias_idx;  // [rows, cols] contributions
	uint rows;
	uint cols;
	float eps;
	uint has_bias;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;
	float n = float(pc.cols);

	float inv_rms = InvRms(heap[pc.x_idx], base, pc.cols, pc.eps, lane);
	GroupMemoryBarrierWithGroupSync();

	// c = sum_i dm[i]*w[i]*x[i],  dm[i] = dy[i]*silu(z[i])
	float local_c = 0.0f;
	for (uint i = lane; i < pc.cols; i += 256u) {
		float zi = OaLoad(heap[pc.z_idx], base + i);
		float gi = Silu(zi);
		float dmi = OaLoad(heap[pc.dy_idx], base + i) * gi;
		local_c += dmi * OaLoad(heap[pc.w_idx], i) * OaLoad(heap[pc.x_idx], base + i);
	}
	g_buf[lane] = local_c;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128u; s > 0u; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float c = g_buf[0];

	for (uint i = lane; i < pc.cols; i += 256u) {
		float xi = OaLoad(heap[pc.x_idx], base + i);
		float wi = OaLoad(heap[pc.w_idx], i);
		float dyi = OaLoad(heap[pc.dy_idx], base + i);
		float zi = OaLoad(heap[pc.z_idx], base + i);
		float sig = Sigmoid(zi);
		float gi = zi * sig;
		float dmi = dyi * gi;

		OaStore(heap[pc.dx_idx], base + i, inv_rms * (dmi * wi - xi * c * inv_rms * inv_rms / n));
		OaStore(heap[pc.dw_idx], base + i, dmi * xi * inv_rms);

		float mi = xi * inv_rms * wi;
		if (pc.has_bias != 0u) {
			mi += OaLoad(heap[pc.bias_idx], i);
			OaStore(heap[pc.dbias_idx], base + i, dmi);
		}
		float dsilu = sig + zi * sig * (1.0f - sig);
		OaStore(heap[pc.dz_idx], base + i, dyi * mi * dsilu);
	}
}
