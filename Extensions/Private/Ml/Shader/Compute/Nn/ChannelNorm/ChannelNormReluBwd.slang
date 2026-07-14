// ChannelNormRelu backward: y = max(0, x_hat * w + b), x_hat = (x - mean) * inv_std
// Same as ChannelNormBwd but masks dyi where the forward output was <= 0 (ReLU).
// Reads the forward output (fwd_out) to determine the ReLU mask. One workgroup
// per (B,T) position. Writes dx, per-row dw_contrib, dbias_contrib.

import Storage;

struct PushConstants {
	uint x_idx;
	uint w_idx;
	uint fwd_out_idx;  // forward output (post-ReLU) for masking
	uint dy_idx;
	uint dx_idx;
	uint dw_contrib_idx;
	uint dbias_contrib_idx;
	uint batch;
	uint channels;
	uint seq_len;
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
	if (row >= pc.batch * pc.seq_len) return;

	uint b = row / pc.seq_len;
	uint t = row % pc.seq_len;
	uint lane = tid.x;
	uint base = b * pc.channels * pc.seq_len + t;
	float n = float(pc.channels);

	// Load assigned values once: x, dy, weight, and the forward ReLU mask. The ReLU
	// mask (fwd_out > 0) is read only once and stored as a bool per element.
	float x_vals[4];
	float dy_vals[4];
	float w_vals[4];
	bool relu_mask[4];
	uint count = 0;
	for (uint c = lane; c < pc.channels; c += 256) {
		uint idx = base + c * pc.seq_len;
		x_vals[count] = OaLoad(heap[pc.x_idx], idx);
		float fwd = OaLoad(heap[pc.fwd_out_idx], idx);
		relu_mask[count] = fwd > 0.0;
		dy_vals[count] = relu_mask[count] ? OaLoad(heap[pc.dy_idx], idx) : 0.0;
		w_vals[count] = OaLoad(heap[pc.w_idx], c);
		++count;
	}

	// Pass 1: mean
	float local_sum = 0.0;
	for (uint i = 0; i < count; ++i) { local_sum += x_vals[i]; }
	float mean = LocalReduceSum256(local_sum, lane) / n;

	GroupMemoryBarrierWithGroupSync();

	// Pass 2: variance -> inv_std
	float local_var = 0.0;
	for (uint i = 0; i < count; ++i) {
		float diff = x_vals[i] - mean;
		local_var += diff * diff;
	}
	float inv_std = rsqrt(LocalReduceSum256(local_var, lane) / n + pc.eps);

	GroupMemoryBarrierWithGroupSync();

	// Pass 3: accumulate dxhat and xhat*dxhat with ReLU-masked dyi
	float local_dxhat = 0.0;
	float local_xhat_dxhat = 0.0;
	for (uint i = 0; i < count; ++i) {
		float x_hat = (x_vals[i] - mean) * inv_std;
		float dyi = dy_vals[i];
		float dxhat = dyi * w_vals[i];
		local_dxhat += dxhat;
		local_xhat_dxhat += x_hat * dxhat;
	}

	float sum_dxhat = LocalReduceSum256(local_dxhat, lane);
	GroupMemoryBarrierWithGroupSync();
	float sum_xhat_dxhat = LocalReduceSum256(local_xhat_dxhat, lane);

	GroupMemoryBarrierWithGroupSync();

	float mean_dxhat = sum_dxhat / n;
	float mean_xhat_dxhat = sum_xhat_dxhat / n;

	// Pass 4: write dx, dw_contrib, dbias_contrib
	uint row_offset = row * pc.channels;
	uint c = lane;
	uint idx = base + c * pc.seq_len;
	for (uint i = 0; i < count; ++i, c += 256, idx += 256 * pc.seq_len) {
		float x_hat = (x_vals[i] - mean) * inv_std;
		float dyi = dy_vals[i];
		float dxhat = dyi * w_vals[i];
		float dx = inv_std * (dxhat - mean_dxhat - x_hat * mean_xhat_dxhat);
		OaStore(heap[pc.dx_idx], idx, dx);
		OaStore(heap[pc.dw_contrib_idx], row_offset + c, dyi * x_hat);
		OaStore(heap[pc.dbias_contrib_idx], row_offset + c, dyi);
	}
}
