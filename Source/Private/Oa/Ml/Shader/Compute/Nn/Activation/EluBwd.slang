// ELU backward pass.
// The autograd tape saves the forward OUTPUT y (not the input x), so we express
// the gradient in terms of y:  d/dx elu = 1 if y > 0 else (y + alpha),
// using alpha*exp(x) == y + alpha for x <= 0 (since y = alpha*(exp(x) - 1)).

import Storage;

struct PushConstants {
	uint y_idx;           // forward output y
	uint grad_out_idx;    // gradient from output
	uint grad_in_idx;     // gradient to input
	uint count;
	float alpha;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;
	
	float y = OaLoad(heap[pc.y_idx], idx);
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx);

	// Gradient factor in terms of the forward output y: 1 if y > 0, else y + alpha.
	float grad_factor = (y > 0.0f) ? 1.0f : (y + pc.alpha);
	float grad_in = grad_out * grad_factor;
	
	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}