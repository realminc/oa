// Mean backward - fused scale + broadcast
// Computes: dInput[i] = dOutput[reduced_idx] / dimSize
// where reduced_idx maps i to the corresponding reduced position

import Storage;

struct PushConstants {
	uint grad_out_idx;   // gradient output (reduced) - buffer index 0
	uint grad_in_idx;    // gradient input (full shape) - buffer index 1
	uint outer_size;     // elements before reduced dim
	uint dim_size;       // size of reduced dimension
	uint inner_size;     // elements after reduced dim
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint total = pc.outer_size * pc.dim_size * pc.inner_size;
	uint idx = tid.x;
	if (idx >= total) return;
	
	// Decompose linear index into (outer, dim, inner)
	uint inner_idx = idx % pc.inner_size;
	uint temp = idx / pc.inner_size;
	uint dim_idx = temp % pc.dim_size;
	uint outer_idx = temp / pc.dim_size;
	
	// Compute reduced index (without dim)
	uint reduced_idx = outer_idx * pc.inner_size + inner_idx;
	
	// Load gradient from reduced tensor, scale by 1/dimSize, store to full tensor
	float grad_out = OaLoad(heap[pc.grad_out_idx], reduced_idx);
	float scale = 1.0f / float(pc.dim_size);
	OaStore(heap[pc.grad_in_idx], idx, grad_out * scale);
}