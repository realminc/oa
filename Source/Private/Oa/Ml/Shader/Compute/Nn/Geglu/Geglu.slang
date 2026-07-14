// Geglu — element-wise GELU on second half, multiply with first half
// Input: [batch, 2*intermediate] (concatenated up and gate for Gemma3)
// Output: [batch, intermediate] (up * GELU(gate))
// Splits input along last dimension, applies GELU to second half, multiplies with first half
// For Gemma3: output = down_proj(GELU(gate) * up)
// We arrange as [up, gate] so: left * GELU(right) = up * GELU(gate) = GELU(gate) * up

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
	float up = OaLoad(heap[pc.in_idx], rowBase + col);
	float gate = OaLoad(heap[pc.in_idx], rowBase + pc.intermediate_size + col);
	
	// Apply GEGLU: up * GELU(gate)
	float gelu_gate = Gelu(gate);
	float result = up * gelu_gate;
	
	OaStore(heap[pc.out_idx], idx, result);
}
