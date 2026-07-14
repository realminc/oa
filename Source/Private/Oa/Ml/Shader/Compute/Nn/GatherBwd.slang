// GatherBwd: scatter-add gradient back into the embedding table.
// Mirrors the forward Gather kernel — supports UInt8 and UInt32 indices.
// Dispatch: one workgroup per vocab row (256 threads cooperate over embed_dim).

import Storage;

struct PushConstants {
	uint indices_idx;
	uint d_out_idx;
	uint d_table_idx;
	uint vocab_size;
	uint embed_dim;
	uint num_indices;
	uint index_dtype;  // 0=UInt8, 1=UInt32 (matches Gather.slang)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.vocab_size) return;
	uint lane = tid.x;

	for (uint d = lane; d < pc.embed_dim; d += 256) {
		float sum = 0.0;
		for (uint i = 0; i < pc.num_indices; i++) {
			uint idx;
			if (pc.index_dtype == 0) {
				idx = (uint)heap[pc.indices_idx].Load<uint8_t>(i);
			} else {
				idx = heap[pc.indices_idx].Load(i * 4);
			}
			if (idx == row) {
				sum += OaLoad(heap[pc.d_out_idx], i * pc.embed_dim + d);
			}
		}
		OaStore(heap[pc.d_table_idx], row * pc.embed_dim + d, sum);
	}
}
