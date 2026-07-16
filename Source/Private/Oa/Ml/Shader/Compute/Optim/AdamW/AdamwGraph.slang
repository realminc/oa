// AdamwGraph — Fused AdamW step with decoupled weight decay, step from buffer.
//
// Identical math to Adamw.slang. The only difference: the optimizer step
// state is read from a host-visible bindless buffer (slot 4) instead of
// living in push constants. This keeps the dispatch's push constants
// invariant across training steps, which is the prerequisite for the
// "compile once, replay many" path (graph.Compile + graph.Replay).
//
// The runtime prepends buffer indices (weights, grads, m, v, state) to push
// constants. A preceding GPU dispatch advances the step word on every replay;
// the host only updates schedule-controlled scalars between completed replays.

import Storage;

struct PushConstants {
	uint weights_idx;    // Stream.cpp prepends: bufs[0]
	uint grads_idx;      // Stream.cpp prepends: bufs[1]
	uint m_idx;          // Stream.cpp prepends: bufs[2]
	uint v_idx;          // Stream.cpp prepends: bufs[3]
	uint state_idx;      // bufs[4]: uint step + five bit-cast float scalars
	uint count;          // Host parameter starts here
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

	uint step = heap[pc.state_idx].Load(0);
	float lr = asfloat(heap[pc.state_idx].Load(4));
	float beta1 = asfloat(heap[pc.state_idx].Load(8));
	float beta2 = asfloat(heap[pc.state_idx].Load(12));
	float eps = asfloat(heap[pc.state_idx].Load(16));
	float weight_decay = asfloat(heap[pc.state_idx].Load(20));

	// Weight decay (decoupled)
	w -= lr * weight_decay * w;

	// Moment updates
	float m_new = beta1 * m_old + (1.0 - beta1) * g;
	float v_new = beta2 * v_old + (1.0 - beta2) * g * g;

	// Bias correction
	float bc1 = 1.0 - pow(beta1, float(step));
	float bc2 = 1.0 - pow(beta2, float(step));
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	// Update
	w -= lr * m_hat / (sqrt(v_hat) + eps);

	OaStore(heap[pc.weights_idx], idx, w);
	OaStore(heap[pc.m_idx], idx, m_new);
	OaStore(heap[pc.v_idx], idx, v_new);
}
