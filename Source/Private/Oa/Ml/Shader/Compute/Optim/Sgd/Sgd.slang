import Storage;

// Buffer indices are prepended by OaVkDispatch::Run (see Runtime/Dispatch.cpp)
// User push constants start AFTER the buffer indices
struct PushConstants {
	uint weights_idx;  // Prepended by infrastructure
	uint grads_idx;    // Prepended by infrastructure
	uint count;        // User parameter
	float lr;
	float weight_decay;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) {
		return;
	}

	float w = OaLoad(heap[pc.weights_idx], idx);
	float g = OaLoad(heap[pc.grads_idx], idx);
	if (pc.weight_decay > 0.0) {
		g += pc.weight_decay * w;
	}
	w -= pc.lr * g;
	OaStore(heap[pc.weights_idx], idx, w);
}
