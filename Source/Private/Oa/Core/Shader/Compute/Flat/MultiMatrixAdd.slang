// MultiMatrixAdd — dst += src for up to 4 tensor pairs in one dispatch.
//
// Buffer indices (dstN_idx, srcN_idx) are auto-prepended by the runtime.
// User push from C++: { count0, count1, count2, count3 }

import Storage;

struct PushConstants {
	// Runtime header: bindless indices for each bound buffer (8 total)
	uint dst0_idx;
	uint src0_idx;
	uint dst1_idx;
	uint src1_idx;
	uint dst2_idx;
	uint src2_idx;
	uint dst3_idx;
	uint src3_idx;
	// User data
	uint count0;
	uint count1;
	uint count2;
	uint count3;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

void AddOne(uint idx, uint count, uint dst_idx, uint src_idx) {
	if (idx >= count) return;
	float a = OaLoad(heap[dst_idx], idx);
	float b = OaLoad(heap[src_idx], idx);
	OaStore(heap[dst_idx], idx, a + b);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	AddOne(idx, pc.count0, pc.dst0_idx, pc.src0_idx);
	AddOne(idx, pc.count1, pc.dst1_idx, pc.src1_idx);
	AddOne(idx, pc.count2, pc.dst2_idx, pc.src2_idx);
	AddOne(idx, pc.count3, pc.dst3_idx, pc.src3_idx);
}
