import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint out_idx;
	uint total;
	uint rank;
	uint out_dims[8];
	uint a_strides[8];
	uint b_strides[8];
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.total) return;

	uint coords[8];
	uint rem = idx;
	for (int d = int(pc.rank) - 1; d >= 0; --d) {
		coords[d] = rem % pc.out_dims[d];
		rem /= pc.out_dims[d];
	}

	uint a_off = 0, b_off = 0;
	for (uint d = 0; d < pc.rank; ++d) {
		a_off += coords[d] * pc.a_strides[d];
		b_off += coords[d] * pc.b_strides[d];
	}
	let a = OaLoad(heap[pc.a_idx], a_off);
	let b = OaLoad(heap[pc.b_idx], b_off);
	OaStore(heap[pc.out_idx], idx, a / b);
}
