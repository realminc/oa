import Storage;

struct PushConstants {
	uint x_idx;          // bufs[0] - Graph.cpp/Stream.cpp adds this
	uint out_idx;        // bufs[1] - Graph.cpp/Stream.cpp adds this
	uint outer_size;     // User parameter
	uint dim_size;       // User parameter
	uint inner_size;     // User parameter
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Reduce one dimension by summing.
// x has shape [outer, dim_size, inner], out has shape [outer, inner].
// Each thread handles one (outer, inner) pair.
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint totalOut = pc.outer_size * pc.inner_size;
	if (idx >= totalOut) return;

	uint inner = idx % pc.inner_size;
	uint outer = idx / pc.inner_size;

	float acc = 0.0;
	uint base = outer * pc.dim_size * pc.inner_size + inner;
	for (uint d = 0; d < pc.dim_size; ++d) {
		acc += OaLoad(heap[pc.x_idx], base + d * pc.inner_size);
	}

	OaStore(heap[pc.out_idx], idx, acc);
}
