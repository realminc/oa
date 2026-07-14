// SmoothL1Mean backward: dA = dLoss * smooth_l1_grad(A - B) / count.
// Element-wise dispatch; one workgroup is enough for typical loss counts, but
// the launch is sized for the full count for generality.

import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint d_out_idx;
	uint d_a_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float a = OaLoad(heap[pc.a_idx], idx);
	float b = OaLoad(heap[pc.b_idx], idx);
	float diff = a - b;
	float absDiff = abs(diff);

	float d_loss = OaLoad(heap[pc.d_out_idx], 0);
	float invN = d_loss / float(pc.count);
	float grad;
	if (absDiff < 1.0f) {
		grad = diff * invN;
	} else {
		grad = (diff >= 0.0f ? 1.0f : -1.0f) * invN;
	}

	OaStore(heap[pc.d_a_idx], idx, grad);
}
