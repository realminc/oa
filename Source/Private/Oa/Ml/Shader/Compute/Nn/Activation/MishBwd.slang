// Mish backward pass
// mish(x) = x * tanh(softplus(x))
// d/dx mish(x) = sech²(softplus(x)) * softplus(x) + mish(x)/x
// Simplified using saved mish output: d/dx ≈ sigmoid(x) * (mish(x) + x * sech²(softplus(x)))

import Storage;
import Activations;  // Sigmoid — shared module (was inline; byte-identical)

struct PushConstants {
	uint x_idx;           // input x
	uint grad_out_idx;    // gradient from output
	uint grad_in_idx;     // gradient to input
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;
	
	float x = OaLoad(heap[pc.x_idx], idx);
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx);
	
	// Recompute softplus for backward pass
	float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
	float tanh_sp = tanh(sp);
	// d/dx softplus(x) = sigmoid(x) — the sigmoid of the INPUT, not of softplus(x).
	float sigmoid_x = Sigmoid(x);  // sigmoid(x) = sp'(x)

	// d/dx mish(x) = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh²(softplus(x)))
	float sech2_sp = 1.0f - tanh_sp * tanh_sp;  // sech²(softplus(x))
	float grad_factor = tanh_sp + x * sigmoid_x * sech2_sp;
	
	float grad_in = grad_out * grad_factor;
	
	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}