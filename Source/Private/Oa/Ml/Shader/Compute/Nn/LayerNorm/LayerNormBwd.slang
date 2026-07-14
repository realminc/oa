// LayerNorm backward: y = x_hat * w + b, x_hat = (x - mean) * inv_std
// dx_hat = dy * w; dx = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(x_hat * dx_hat))
// dw row contribution: dy * x_hat (column-sum via ReduceCols on host)
// Inlined reductions (same pattern as LayerNorm.slang).

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

float LocalReduceSum256(float val, uint lane) {
	g_buf[lane] = val;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lane < s) {
			g_buf[lane] += g_buf[lane + s];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return g_buf[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;
	float n = float(pc.cols);

	float local_sum = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		local_sum += OaLoad(heap[pc.x_idx], base + i);
	}
	float mean = LocalReduceSum256(local_sum, lane) / n;

	GroupMemoryBarrierWithGroupSync();

	float local_var = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		float diff = OaLoad(heap[pc.x_idx], base + i) - mean;
		local_var += diff * diff;
	}
	float inv_std = rsqrt(LocalReduceSum256(local_var, lane) / n + pc.eps);

	GroupMemoryBarrierWithGroupSync();

	float local_dxhat = 0.0;
	float local_xhat_dxhat = 0.0;
	for (uint i = lane; i < pc.cols; i += 256) {
		float xi = OaLoad(heap[pc.x_idx], base + i);
		float x_hat = (xi - mean) * inv_std;
		float dyi = OaLoad(heap[pc.dy_idx], base + i);
		float wi = OaLoad(heap[pc.w_idx], i);
		float dxhat = dyi * wi;
		local_dxhat += dxhat;
		local_xhat_dxhat += x_hat * dxhat;
	}

	float sum_dxhat = LocalReduceSum256(local_dxhat, lane);
	GroupMemoryBarrierWithGroupSync();
	float sum_xhat_dxhat = LocalReduceSum256(local_xhat_dxhat, lane);

	GroupMemoryBarrierWithGroupSync();

	float mean_dxhat = sum_dxhat / n;
	float mean_xhat_dxhat = sum_xhat_dxhat / n;

	for (uint i = lane; i < pc.cols; i += 256) {
		float xi = OaLoad(heap[pc.x_idx], base + i);
		float x_hat = (xi - mean) * inv_std;
		float dyi = OaLoad(heap[pc.dy_idx], base + i);
		float wi = OaLoad(heap[pc.w_idx], i);
		float dxhat = dyi * wi;
		float dx = inv_std * (dxhat - mean_dxhat - x_hat * mean_xhat_dxhat);
		OaStore(heap[pc.dx_idx], base + i, dx);
		OaStore(heap[pc.dw_contrib_idx], base + i, dyi * x_hat);
	}
}
