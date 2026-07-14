// SiluAndMul — element-wise SiLU on first half, multiply with second half
// Input: [batch, 2*intermediate] (concatenated gate and up)
// Output: [batch, intermediate] (SiLU(gate) * up)
// Splits input along last dimension, applies SiLU to first half, multiplies with second half

import Activations;
import Storage;

struct PushConstants {
	uint in_idx;
	uint out_idx;
	uint count;
	uint intermediate_size;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint idx = gid.x * 256 + lid.x;
	
	if (idx >= pc.count) return;
	
	uint row = idx / pc.intermediate_size;
	uint col = idx % pc.intermediate_size;
	uint rowBase = row * (2 * pc.intermediate_size);
	float gate = OaLoad(heap[pc.in_idx], rowBase + col);
	float up = OaLoad(heap[pc.in_idx], rowBase + pc.intermediate_size + col);
	
	// Apply SwiGLU: SiLU(gate) * up
	float result = Swiglu(gate, up);
	
	OaStore(heap[pc.out_idx], idx, result);
}
