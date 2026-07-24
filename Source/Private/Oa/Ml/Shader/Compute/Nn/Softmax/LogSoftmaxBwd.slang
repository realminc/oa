// Log-softmax backward: dx_i = dy_i - exp(y_i) * sum(dy).

import Storage;

struct PushConstants {
	uint forward_output_idx;
	uint grad_output_idx;
	uint grad_input_idx;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Ping-pong banks keep each reduction pass's reads and writes at distinct
// shared addresses. This avoids same-address RMW ambiguity in older GPU-AV
// instrumentation while preserving the barrier-separated reduction tree.
groupshared float reduction[2][256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint axisGroup = group.x;
	uint lane = thread.x;
	if (axisGroup >= pc.outer_size * pc.inner_size) return;
	uint outer = axisGroup / pc.inner_size;
	uint inner = axisGroup % pc.inner_size;
	uint base = outer * pc.dim_size * pc.inner_size + inner;
	uint readBank = 0u;
	uint writeBank = 1u;

	float localSum = 0.0f;
	for (uint column = lane; column < pc.dim_size; column += 256u) {
		localSum += OaLoad(
			heap[pc.grad_output_idx], base + column * pc.inner_size);
	}
	reduction[readBank][lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[writeBank][lane] =
			reduction[readBank][lane] + reduction[readBank][lane + width];
		GroupMemoryBarrierWithGroupSync();
		uint previousReadBank = readBank;
		readBank = writeBank;
		writeBank = previousReadBank;
	}
	float gradientSum = reduction[readBank][0];
	for (uint column = lane; column < pc.dim_size; column += 256u) {
		uint index = base + column * pc.inner_size;
		float output = OaLoad(heap[pc.forward_output_idx], index);
		float upstream = OaLoad(heap[pc.grad_output_idx], index);
		OaStore(heap[pc.grad_input_idx], index,
			upstream - exp(output) * gradientSum);
	}
}
