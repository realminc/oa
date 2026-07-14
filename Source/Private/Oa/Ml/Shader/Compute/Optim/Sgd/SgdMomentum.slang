import Storage;

// Fused SGD with momentum:
//   momentum = momentum_coef * momentum + grad
//   param -= lr * momentum + weight_decay * param
struct PushConstants {
	uint weights_idx;
	uint grads_idx;
	uint momentum_idx;
	uint count;
	float lr;
	float momentum_coef;
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
	float m = OaLoad(heap[pc.momentum_idx], idx);

	m = pc.momentum_coef * m + g;
	OaStore(heap[pc.momentum_idx], idx, m);

	if (pc.weight_decay > 0.0f) {
		w -= pc.lr * m + pc.weight_decay * w;
	} else {
		w -= pc.lr * m;
	}
	OaStore(heap[pc.weights_idx], idx, w);
}
