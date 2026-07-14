// Softplus backward pass
// d/da softplus(a) = sigmoid(a). We save the forward output y = softplus(a) ≥ 0,
// and sigmoid(a) = 1 - e^-y (exact: e^y = 1 + e^a ⇒ e^a/(1+e^a) = 1 - e^-y).
// This is numerically stable because y ≥ 0 ⇒ e^-y ∈ (0, 1].
// So: dA = dOut * (1 - e^-y).

import Storage;

struct PushConstants {
	uint softplus_y_idx;
	uint grad_out_idx;
	uint grad_in_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	float y        = OaLoad(heap[pc.softplus_y_idx], idx);  // Saved softplus(a) output
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx);    // Incoming gradient

	float grad_in = grad_out * (1.0f - exp(-y));             // dOut * sigmoid(a)

	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}
