// RMSNorm backward: dx = inv_rms * (dy*w - x * c * inv_rms^2 / n), dw_contrib = dy * x * inv_rms
// Uses InvRms from reductions (stays within module — works correctly).
// Direct ReduceSum256 call replaced with inlined reduction to avoid cross-module groupshared bug.

import Reductions;
import Storage;

struct PushConstants {
	uint x_idx;
	uint w_idx;
	uint dy_idx;
	uint dx_idx;
	uint dw_contrib_idx;
	uint rows;
	uint cols;
	float eps;
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

	float local_c = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		local_c += OaLoad(heap[pc.dy_idx], base + i) * OaLoad(heap[pc.w_idx], i) * OaLoad(heap[pc.x_idx], base + i);
	}

	g_buf[lane] = local_c;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) g_buf[lane] += g_buf[lane + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float c = g_buf[0];

	for (uint i = lane; i < pc.cols; i += 256) {
		float xi = OaLoad(heap[pc.x_idx], base + i);
		float dyi = OaLoad(heap[pc.dy_idx], base + i);
		float wi = OaLoad(heap[pc.w_idx], i);

		OaStore(heap[pc.dx_idx], base + i, inv_rms * (dyi * wi - xi * c * inv_rms * inv_rms / n));
		OaStore(heap[pc.dw_contrib_idx], base + i, dyi * xi * inv_rms);
	}
}
