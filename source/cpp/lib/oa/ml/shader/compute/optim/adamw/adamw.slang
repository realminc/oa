// Adamw — Fused AdamW optimizer step with decoupled weight decay
// Stream.cpp prepends buffer indices (weights, grads, m, v) to push constants.
// Host passes: count, lr, beta1, beta2, eps, weight_decay, step

import storage;

struct PushConstants {
	uint weights_idx;    // Stream.cpp prepends: bufs[0]
	uint grads_idx;      // Stream.cpp prepends: bufs[1]
	uint m_idx;          // Stream.cpp prepends: bufs[2]
	uint v_idx;          // Stream.cpp prepends: bufs[3]
	uint count;          // Host parameter starts here
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
	uint step;           // 1-indexed (for bias correction)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float w = oaLoad(heap[pc.weights_idx], idx);
	float g = oaLoad(heap[pc.grads_idx], idx);
	float m_old = oaLoad(heap[pc.m_idx], idx);
	float v_old = oaLoad(heap[pc.v_idx], idx);

	// Weight decay (decoupled)
	w -= pc.lr * pc.weight_decay * w;

	// Moment updates
	float m_new = pc.beta1 * m_old + (1.0 - pc.beta1) * g;
	float v_new = pc.beta2 * v_old + (1.0 - pc.beta2) * g * g;

	// Bias correction
	float bc1 = 1.0 - pow(pc.beta1, float(pc.step));
	float bc2 = 1.0 - pow(pc.beta2, float(pc.step));
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	// Update
	w -= pc.lr * m_hat / (sqrt(v_hat) + pc.eps);

	oaStore(heap[pc.weights_idx], idx, w);
	oaStore(heap[pc.m_idx], idx, m_new);
	oaStore(heap[pc.v_idx], idx, v_new);
}
