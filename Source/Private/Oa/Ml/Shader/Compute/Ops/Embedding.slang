// General embedding lookup: out[i, :] = table[indices[i], :]
// Same as gather but with explicit naming for the embedding use case.

import Storage;

struct PushConstants {
	uint table_idx;
	uint indices_idx;
	uint out_idx;
	uint numIndices;
	uint dim;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.numIndices * pc.dim) return;

	uint row = idx / pc.dim;
	uint col = idx % pc.dim;
	uint tableRow = heap[pc.indices_idx].Load(row * 4);
	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.table_idx], tableRow * pc.dim + col));
}
