// MultiMatrixFill — fill up to 4 tensors with a constant value in one dispatch.
//
// Buffer indices (bufN_idx) are auto-prepended by the runtime; they are NOT
// part of the C++ push struct. The shader struct lists them first so the
// combined push-constant block laid out by the runtime matches.
//
// User push from C++: { count0, count1, count2, count3, Value }

import Storage;

struct PushConstants {
	// Runtime header: bindless indices for each bound buffer
	uint buf0_idx;
	uint buf1_idx;
	uint buf2_idx;
	uint buf3_idx;
	// User data
	uint count0;
	uint count1;
	uint count2;
	uint count3;
	float Value;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

void FillOne(uint idx, uint count, uint buf_idx) {
	if (idx >= count) return;
	OaStore(heap[buf_idx], idx, pc.Value);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	FillOne(idx, pc.count0, pc.buf0_idx);
	FillOne(idx, pc.count1, pc.buf1_idx);
	FillOne(idx, pc.count2, pc.buf2_idx);
	FillOne(idx, pc.count3, pc.buf3_idx);
}
