// VelSmoothL1 backward: gradient w.r.t. the prediction tensor [B, T, D].
// Velocity is computed along the time dimension. For each frame t, the prediction
// contributes to at most two velocity terms: (t-1, t) and (t, t+1). The kernel
// writes dPred[b, t, d] using the finite-difference chain rule.

import Storage;

struct PushConstants {
	uint pred_idx;
	uint target_idx;
	uint d_out_idx;
	uint d_pred_idx;
	uint batch;
	uint seq_len;
	uint feat_dim;
	uint count;    // batch * (seq_len - 1) * feat_dim
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float SmoothL1Grad(float diff) {
	float absDiff = abs(diff);
	if (absDiff < 1.0f) {
		return diff;
	} else {
		return diff >= 0.0f ? 1.0f : -1.0f;
	}
}

float Load3D(uint buf_idx, uint b, uint t, uint d) {
	uint idx = (b * pc.seq_len + t) * pc.feat_dim + d;
	return OaLoad(heap[buf_idx], idx);
}

void Store3D(uint buf_idx, uint b, uint t, uint d, float val) {
	uint idx = (b * pc.seq_len + t) * pc.feat_dim + d;
	OaStore(heap[buf_idx], idx, val);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint flat = tid.x;
	const uint N = pc.batch * pc.seq_len * pc.feat_dim;
	if (flat >= N) return;

	const uint D = pc.feat_dim;
	const uint T = pc.seq_len;
	uint b = flat / (T * D);
	uint rem = flat % (T * D);
	uint t = rem / D;
	uint d = rem % D;

	float d_loss = OaLoad(heap[pc.d_out_idx], 0);
	float invN = d_loss / float(pc.count);
	float grad = 0.0f;

	// Contribution as the right endpoint of the (t-1, t) velocity pair.
	if (t > 0) {
		float p_diff = Load3D(pc.pred_idx, b, t, d) - Load3D(pc.pred_idx, b, t - 1, d);
		float t_diff = Load3D(pc.target_idx, b, t, d) - Load3D(pc.target_idx, b, t - 1, d);
		grad += SmoothL1Grad(p_diff - t_diff) * invN;
	}

	// Contribution as the left endpoint of the (t, t+1) velocity pair.
	if (t + 1 < T) {
		float p_diff = Load3D(pc.pred_idx, b, t + 1, d) - Load3D(pc.pred_idx, b, t, d);
		float t_diff = Load3D(pc.target_idx, b, t + 1, d) - Load3D(pc.target_idx, b, t, d);
		grad -= SmoothL1Grad(p_diff - t_diff) * invN;
	}

	Store3D(pc.d_pred_idx, b, t, d, grad);
}
