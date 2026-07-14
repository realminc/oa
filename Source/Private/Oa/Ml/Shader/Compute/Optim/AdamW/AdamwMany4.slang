// AdamwMany4 — fused AdamW step for four parameter tensors.
// Buffer order per tensor: weights, grads, m, v.

import Storage;

struct PushConstants {
	uint w0_idx; uint g0_idx; uint m0_idx; uint v0_idx;
	uint w1_idx; uint g1_idx; uint m1_idx; uint v1_idx;
	uint w2_idx; uint g2_idx; uint m2_idx; uint v2_idx;
	uint w3_idx; uint g3_idx; uint m3_idx; uint v3_idx;
	uint count0;
	uint count1;
	uint count2;
	uint count3;
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	uint step;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

void UpdateOne(uint idx, uint count, uint w_idx, uint g_idx, uint m_idx, uint v_idx) {
	if (idx >= count) return;

	float w = OaLoad(heap[w_idx], idx);
	float g = OaLoad(heap[g_idx], idx);
	float m_old = OaLoad(heap[m_idx], idx);
	float v_old = OaLoad(heap[v_idx], idx);

	w -= pc.lr * pc.weight_decay * w;

	float m_new = pc.beta1 * m_old + (1.0 - pc.beta1) * g;
	float v_new = pc.beta2 * v_old + (1.0 - pc.beta2) * g * g;

	float bc1 = 1.0 - pow(pc.beta1, float(pc.step));
	float bc2 = 1.0 - pow(pc.beta2, float(pc.step));
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	w -= pc.lr * m_hat / (sqrt(v_hat) + pc.eps);

	OaStore(heap[w_idx], idx, w);
	OaStore(heap[m_idx], idx, m_new);
	OaStore(heap[v_idx], idx, v_new);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	UpdateOne(idx, pc.count0, pc.w0_idx, pc.g0_idx, pc.m0_idx, pc.v0_idx);
	UpdateOne(idx, pc.count1, pc.w1_idx, pc.g1_idx, pc.m1_idx, pc.v1_idx);
	UpdateOne(idx, pc.count2, pc.w2_idx, pc.g2_idx, pc.m2_idx, pc.v2_idx);
	UpdateOne(idx, pc.count3, pc.w3_idx, pc.g3_idx, pc.m3_idx, pc.v3_idx);
}
