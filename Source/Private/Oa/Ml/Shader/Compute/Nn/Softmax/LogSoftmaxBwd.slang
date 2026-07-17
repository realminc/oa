// Log-softmax backward: dx_i = dy_i - exp(y_i) * sum(dy).

import Storage;

struct PushConstants {
	uint forward_output_idx;
	uint grad_output_idx;
	uint grad_input_idx;
	uint rows;
	uint cols;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float reduction[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
	uint row = group.x;
	uint lane = thread.x;
	if (row >= pc.rows) return;
	uint base = row * pc.cols;

	float localSum = 0.0f;
	for (uint column = lane; column < pc.cols; column += 256u) {
		localSum += OaLoad(heap[pc.grad_output_idx], base + column);
	}
	reduction[lane] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128u; width > 0u; width >>= 1u) {
		if (lane < width) reduction[lane] += reduction[lane + width];
		GroupMemoryBarrierWithGroupSync();
	}
	float gradientSum = reduction[0];
	for (uint column = lane; column < pc.cols; column += 256u) {
		float output = OaLoad(heap[pc.forward_output_idx], base + column);
		float upstream = OaLoad(heap[pc.grad_output_idx], base + column);
		OaStore(heap[pc.grad_input_idx], base + column,
			upstream - exp(output) * gradientSum);
	}
}
