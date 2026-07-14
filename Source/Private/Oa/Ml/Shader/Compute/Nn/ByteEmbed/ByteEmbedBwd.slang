import Storage;

struct PushConstants {
	uint d_out_idx;
	uint indices_idx;
	uint d_table_idx;
	uint count;
	uint dim;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint vocab_idx = gid.x;
	if (vocab_idx >= 256) return;

	uint lane = tid.x;

	for (uint d = lane; d < pc.dim; d += 256) {
		float sum = 0.0;
		for (uint t = 0; t < pc.count; t++) {
			if (heap[pc.indices_idx].Load(t * 4) == vocab_idx) {
				sum += OaLoad(heap[pc.d_out_idx], t * pc.dim + d);
			}
		}
		OaStore(heap[pc.d_table_idx], vocab_idx * pc.dim + d, sum);
	}
}
