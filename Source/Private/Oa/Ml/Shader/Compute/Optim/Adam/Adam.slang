import Storage;

// Buffer indices are prepended by OaVkDispatch::Run (see Runtime/Dispatch.cpp)
// User push constants start AFTER the buffer indices
struct PushConstants {
	uint weights_idx;  // Prepended by infrastructure
	uint grads_idx;    // Prepended by infrastructure
	uint m_idx;        // Prepended by infrastructure
	uint v_idx;        // Prepended by infrastructure
	uint count;        // User parameter
	float lr;
	float beta1;
	float beta2;
	float eps;
	uint step;
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

	float m_new = pc.beta1 * OaLoad(heap[pc.m_idx], idx) + (1.0 - pc.beta1) * g;
	float v_new = pc.beta2 * OaLoad(heap[pc.v_idx], idx) + (1.0 - pc.beta2) * g * g;

	float bc1 = 1.0 - pow(pc.beta1, float(pc.step));
	float bc2 = 1.0 - pow(pc.beta2, float(pc.step));
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	w -= pc.lr * m_hat / (sqrt(v_hat) + pc.eps);

	OaStore(heap[pc.weights_idx], idx, w);
	OaStore(heap[pc.m_idx], idx, m_new);
	OaStore(heap[pc.v_idx], idx, v_new);
}
