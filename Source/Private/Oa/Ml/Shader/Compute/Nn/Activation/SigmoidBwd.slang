// Sigmoid backward pass
// d/dx σ(x) = σ(x) * (1 - σ(x))
// We save σ(x) as output, so: dX = dOut * σ(x) * (1 - σ(x))

import Storage;

struct PushConstants {
	uint sigmoid_x_idx;
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
	
	float sigmoid_x = OaLoad(heap[pc.sigmoid_x_idx], idx);   // Saved σ(x) output
	float grad_out = OaLoad(heap[pc.grad_out_idx], idx);     // Incoming gradient
	
	// Gradient: dOut * σ(x) * (1 - σ(x))
	float grad_in = grad_out * sigmoid_x * (1.0f - sigmoid_x);
	
	OaStore(heap[pc.grad_in_idx], idx, grad_in);
}