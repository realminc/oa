// GreaterEqual — GPU-native elementwise scalar comparison, producing an FP32 mask.

import Storage;

struct PushConstants {
	uint input_idx;
	uint output_idx;
	uint n;
	float value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.n) return;
	float result = OaLoad(heap[pc.input_idx], i) >= pc.value ? 1.0f : 0.0f;
	heap[pc.output_idx].Store(i * 4, asuint(result));
}
