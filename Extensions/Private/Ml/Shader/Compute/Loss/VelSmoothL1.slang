// VelSmoothL1 forward: velocity SmoothL1 loss reduced to a scalar mean in one pass.
// Computes finite-difference velocity along the time dimension, applies SmoothL1,
// and reduces. Shapes: pred [B, T, D], target [B, T, D].
// Output: out[0] = mean(SmoothL1(vel_pred - vel_target)).

import Storage;

struct PushConstants {
	uint pred_idx;
	uint target_idx;
	uint out_idx;
	uint batch;
	uint seq_len;
	uint feat_dim;
	uint count;  // batch * (seq_len - 1) * feat_dim
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_buf[256];

float SmoothL1(float diff) {
	float absDiff = abs(diff);
	if (absDiff < 1.0f) {
		return 0.5f * diff * diff;
	} else {
		return absDiff - 0.5f;
	}
}

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

float Load3D(uint buf_idx, uint b, uint t, uint d) {
	uint idx = (b * pc.seq_len + t) * pc.feat_dim + d;
	return OaLoad(heap[buf_idx], idx);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint lane = tid.x;
	const uint T = pc.seq_len;
	const uint D = pc.feat_dim;
	const uint velT = T - 1;

	float local_sum = 0.0f;
	for (uint flat = lane; flat < pc.count; flat += 256) {
		uint b = flat / (velT * D);
		uint rem = flat % (velT * D);
		uint t = rem / D;
		uint d = rem % D;

		float p_next = Load3D(pc.pred_idx, b, t + 1, d);
		float p_cur  = Load3D(pc.pred_idx, b, t, d);
		float t_next = Load3D(pc.target_idx, b, t + 1, d);
		float t_cur  = Load3D(pc.target_idx, b, t, d);

		float velDiff = (p_next - p_cur) - (t_next - t_cur);
		local_sum += SmoothL1(velDiff);
	}

	float total = LocalReduceSum256(local_sum, lane);
	if (lane == 0) {
		OaStore(heap[pc.out_idx], 0, total / float(pc.count));
	}
}
