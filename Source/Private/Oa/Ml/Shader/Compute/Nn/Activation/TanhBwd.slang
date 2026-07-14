// Tanh backward pass
// d/dx tanh(x) = 1 - tanh²(x)
// We save tanh(x) as output, so: dX = dOut * (1 - tanh²(x))

import Storage;

struct PushConstants {
	uint tanh_x_idx;
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
	
	float tanh_x = OaLoad(heap[pc.tanh_x_idx], idx);      // Saved tanh(x) output
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx);  // Incoming gradient
	
	// Gradient: dOut * (1 - tanh²(x))
	float grad_in = grad_out * (1.0f - tanh_x * tanh_x);
	
	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}