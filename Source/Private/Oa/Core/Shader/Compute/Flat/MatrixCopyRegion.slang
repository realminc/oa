// Copy a dense rank-1..4 matrix into a same-rank matrix while applying one
// source and destination offset. Slice uses SrcStart; Concat uses DstStart.

import Storage;

struct PushConstants {
	uint src_idx;
	uint dst_idx;
	uint count;
	uint rank;
	uint dim;
	uint src_start;
	uint dst_start;
	uint src_dim0;
	uint src_dim1;
	uint src_dim2;
	uint src_dim3;
	uint dst_dim0;
	uint dst_dim1;
	uint dst_dim2;
	uint dst_dim3;
	uint copy_dim0;
	uint copy_dim1;
	uint copy_dim2;
	uint copy_dim3;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint src_dims[4] = {pc.src_dim0, pc.src_dim1, pc.src_dim2, pc.src_dim3};
	uint dst_dims[4] = {pc.dst_dim0, pc.dst_dim1, pc.dst_dim2, pc.dst_dim3};
	uint copy_dims[4] = {pc.copy_dim0, pc.copy_dim1, pc.copy_dim2, pc.copy_dim3};
	uint coords[4] = {0, 0, 0, 0};

	uint remaining = idx;
	for (int d = int(pc.rank) - 1; d >= 0; --d) {
		coords[d] = remaining % copy_dims[d];
		remaining /= copy_dims[d];
	}

	uint src_flat = 0;
	uint dst_flat = 0;
	for (uint d = 0; d < pc.rank; ++d) {
		uint src_coord = coords[d] + ((d == pc.dim) ? pc.src_start : 0);
		uint dst_coord = coords[d] + ((d == pc.dim) ? pc.dst_start : 0);
		src_flat = src_flat * src_dims[d] + src_coord;
		dst_flat = dst_flat * dst_dims[d] + dst_coord;
	}

	OaStore(heap[pc.dst_idx], dst_flat, OaLoad(heap[pc.src_idx], src_flat));
}
