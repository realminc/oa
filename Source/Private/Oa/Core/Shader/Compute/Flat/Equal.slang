// Equal — GPU-native elementwise scalar equality.
// Output is always FP32 0/1 because it is used as a mask by backward graphs.

import Storage;

struct PushConstants {
	uint input_idx;
	uint output_idx;
	uint n;
	uint inputType; // 0 = floating storage, 1 = int32
	float value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.n) return;
	bool equal;
	if (pc.inputType == 1) {
		equal = asint(heap[pc.input_idx].Load(i * 4)) == int(pc.value);
	} else {
		equal = OaLoad(heap[pc.input_idx], i) == pc.value;
	}
	heap[pc.output_idx].Store(i * 4, asuint(equal ? 1.0f : 0.0f));
}
