import Storage;

struct PushConstants {
	uint src_idx;
	uint out_idx;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Broadcast a reduced tensor back to its original shape along one dimension.
// src has shape [outer, inner], out has shape [outer, dim_size, inner].
// For global broadcast (scalar to full shape): outer=1, dim_size=total, inner=1.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.outer_size * pc.dim_size * pc.inner_size;
	if (idx >= total) return;

	uint inner = idx % pc.inner_size;
	uint dim = (idx / pc.inner_size) % pc.dim_size;
	uint outer = idx / (pc.dim_size * pc.inner_size);

	uint srcIdx = outer * pc.inner_size + inner;
	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.src_idx], srcIdx));
}
