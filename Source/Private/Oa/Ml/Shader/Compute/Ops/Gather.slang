// Table lookup: out[i] = table[indices[i] * stride ... + stride]
// Each thread gathers one row.
// Supports both UInt8 (byte embedding) and UInt32 (LLM tokens) indices.

import Storage;

struct PushConstants {
	uint table_idx;
	uint indices_idx;
	uint out_idx;
	uint numIndices;
	uint rowSize;
	uint index_dtype;  // 0=UInt8, 1=UInt32
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.numIndices * pc.rowSize) return;

	uint row = idx / pc.rowSize;
	uint col = idx % pc.rowSize;
	
	// Load index based on dtype (UInt8 or UInt32)
	uint tableRow;
	if (pc.index_dtype == 0) {
		// UInt8: 1 byte per index
		tableRow = (uint)heap[pc.indices_idx].Load<uint8_t>(row);
	} else {
		// UInt32: 4 bytes per index
		tableRow = heap[pc.indices_idx].Load(row * 4);
	}
	
	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.table_idx], tableRow * pc.rowSize + col));
}
