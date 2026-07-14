// LeakyReLU backward pass
// d/dx leaky_relu(x, α) = x > 0 ? 1 : α
// We save input x for backward

import Storage;

struct PushConstants {
	uint x_idx;
	uint grad_out_idx;
	uint grad_in_idx;
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
	
	float x = OaLoad(heap[pc.x_idx], idx);              // Saved input
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx); // Incoming gradient
	
	// Gradient: x > 0 ? dOut : dOut * α
	float grad_in = (x > 0.0f) ? grad_out : (grad_out * pc.alpha);
	
	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}