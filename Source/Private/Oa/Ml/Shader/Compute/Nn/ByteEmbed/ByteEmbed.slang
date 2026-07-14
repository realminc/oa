import Storage;

struct PushConstants {
	uint indices_idx;
	uint table_idx;
	uint out_idx;
	uint count;
	uint dim;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count * pc.dim) return;

	uint token = idx / pc.dim;
	uint d     = idx % pc.dim;
	uint byte_val = heap[pc.indices_idx].Load(token * 4);
	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.table_idx], byte_val * pc.dim + d));
}
