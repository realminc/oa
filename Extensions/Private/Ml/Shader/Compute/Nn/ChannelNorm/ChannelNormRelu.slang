// ChannelNormRelu: fused ChannelNorm + ReLU on [B,C,T] tensors.
// Same as ChannelNorm but clamps output to >= 0 (ReLU). Saves 1 dispatch per
// site vs ChannelNorm + Relu. One workgroup per (B,T) position; 256 threads
// reduce over C.

import Storage;

struct PushConstants {
	uint x_idx;
	uint weight_idx;
	uint bias_idx;
	uint out_idx;
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

	// Load all assigned elements once into registers; the strided [B,C,T] layout
	// makes each read a separate cache line, so re-reading for mean/variance/write
	// was the dominant cost. With values in registers we do one global load per element.
	float vals[4];
	uint count = 0;
	for (uint c = lane; c < pc.channels; c += 256) {
		vals[count++] = OaLoad(heap[pc.x_idx], base + c * pc.seq_len);
	}

	// Pass 1: mean
	float local_sum = 0.0;
	for (uint i = 0; i < count; ++i) { local_sum += vals[i]; }
	float mean = LocalReduceSum256(local_sum, lane) / n;

	GroupMemoryBarrierWithGroupSync();

	// Pass 2: variance
	float local_var = 0.0;
	for (uint i = 0; i < count; ++i) {
		float diff = vals[i] - mean;
		local_var += diff * diff;
	}
	float inv_std = rsqrt(LocalReduceSum256(local_var, lane) / n + pc.eps);

	// Pass 3: normalize, apply affine, then ReLU
	uint c = lane;
	uint idx = base + c * pc.seq_len;
	for (uint i = 0; i < count; ++i, c += 256, idx += 256 * pc.seq_len) {
		float normed = (vals[i] - mean) * inv_std;
		float y = normed * OaLoad(heap[pc.weight_idx], c) + OaLoad(heap[pc.bias_idx], c);
		OaStore(heap[pc.out_idx], idx, max(0.0, y));
	}
}
