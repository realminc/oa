// AdamwGraph — Fused AdamW step with decoupled weight decay, step from buffer.
//
// Identical math to Adamw.slang. The only difference: the optimizer step
// counter is read from a host-visible bindless buffer (slot 4) instead of
// living in push constants. This keeps the dispatch's push constants
// invariant across training steps, which is the prerequisite for the
// "compile once, replay many" path (graph.Compile + graph.Replay).
//
// Stream.cpp prepends buffer indices (weights, grads, m, v, step) to push
// constants. The host writes a single float (step number as float) into
// the step buffer before each training step.

import Storage;

struct PushConstants {
	uint weights_idx;    // Stream.cpp prepends: bufs[0]
	uint grads_idx;      // Stream.cpp prepends: bufs[1]
	uint m_idx;          // Stream.cpp prepends: bufs[2]
	uint v_idx;          // Stream.cpp prepends: bufs[3]
	uint step_idx;       // Stream.cpp prepends: bufs[4]  (host-visible, 1 float)
	uint count;          // Host parameter starts here
	float lr;
	float beta1;
	float beta2;
	float eps;
	float weight_decay;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float w = OaLoad(heap[pc.weights_idx], idx);
	float g = OaLoad(heap[pc.grads_idx], idx);
	float m_old = OaLoad(heap[pc.m_idx], idx);
	float v_old = OaLoad(heap[pc.v_idx], idx);

	// Step number read from host-written buffer (slot 0 holds the current step).
	float step_f = OaLoad(heap[pc.step_idx], 0);

	// Weight decay (decoupled)
	w -= pc.lr * pc.weight_decay * w;

	// Moment updates
	float m_new = pc.beta1 * m_old + (1.0 - pc.beta1) * g;
	float v_new = pc.beta2 * v_old + (1.0 - pc.beta2) * g * g;

	// Bias correction
	float bc1 = 1.0 - pow(pc.beta1, step_f);
	float bc2 = 1.0 - pow(pc.beta2, step_f);
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	// Update
	w -= pc.lr * m_hat / (sqrt(v_hat) + pc.eps);

	OaStore(heap[pc.weights_idx], idx, w);
	OaStore(heap[pc.m_idx], idx, m_new);
	OaStore(heap[pc.v_idx], idx, v_new);
}
