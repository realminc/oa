import Storage;

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint rows;
	uint cols;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float sdata[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	uint lid = tid.x;
	if (row >= pc.rows) return;

	uint base = row * pc.cols;

	float localMax = -3.402823466e+38f;
	for (uint i = lid; i < pc.cols; i += 256) {
		localMax = max(localMax, OaLoad(heap[pc.x_idx], base + i));
	}
	sdata[lid] = localMax;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) sdata[lid] = max(sdata[lid], sdata[lid + s]);
		GroupMemoryBarrierWithGroupSync();
	}
	float rowMax = sdata[0];
	GroupMemoryBarrierWithGroupSync();

	float localSum = 0.0;
	for (uint i = lid; i < pc.cols; i += 256) {
		localSum += exp(OaLoad(heap[pc.x_idx], base + i) - rowMax);
	}
	sdata[lid] = localSum;
	GroupMemoryBarrierWithGroupSync();
	for (uint s = 128; s > 0; s >>= 1) {
		if (lid < s) sdata[lid] += sdata[lid + s];
		GroupMemoryBarrierWithGroupSync();
	}
	float invSum = 1.0 / sdata[0];
	GroupMemoryBarrierWithGroupSync();

	for (uint i = lid; i < pc.cols; i += 256) {
		OaStore(heap[pc.out_idx], base + i, exp(OaLoad(heap[pc.x_idx], base + i) - rowMax) * invSum);
	}
}
