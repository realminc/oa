// Two/three independent projections of one activation in a single dispatch.
import Storage;
struct PushConstants {
	uint x_idx; uint w0_idx; uint w1_idx; uint w2_idx;
	uint b0_idx; uint b1_idx; uint b2_idx; uint out_idx;
	uint M; uint K; uint N0; uint N1; uint N2; uint Count; uint HasBias;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint width = pc.N0 + pc.N1 + (pc.Count == 3 ? pc.N2 : 0u);
	uint linear = tid.x;
	if (linear >= pc.M * width) return;
	uint m = linear / width, packedN = linear % width;
	float sum = 0.0;
	// Keep every descriptor-array index dynamically uniform. Selecting w/b into
	// thread-varying local variables makes the descriptor itself non-uniform as
	// adjacent lanes cross a packed projection boundary. Some desktop drivers
	// tolerated the missing NonUniform decoration; Turnip correctly exposed it
	// as corrupted Q/K/V projections. The explicit branches retain one dispatch
	// while each heap index remains a uniform push-constant expression.
	if (packedN < pc.N0) {
		uint n = packedN;
		sum = pc.HasBias != 0 ? OaLoad(heap[pc.b0_idx], n) : 0.0;
		for (uint k = 0; k < pc.K; ++k)
			sum += OaLoad(heap[pc.x_idx], m * pc.K + k) *
				OaLoad(heap[pc.w0_idx], n * pc.K + k);
	} else if (packedN < pc.N0 + pc.N1) {
		uint n = packedN - pc.N0;
		sum = pc.HasBias != 0 ? OaLoad(heap[pc.b1_idx], n) : 0.0;
		for (uint k = 0; k < pc.K; ++k)
			sum += OaLoad(heap[pc.x_idx], m * pc.K + k) *
				OaLoad(heap[pc.w1_idx], n * pc.K + k);
	} else {
		uint n = packedN - pc.N0 - pc.N1;
		sum = pc.HasBias != 0 ? OaLoad(heap[pc.b2_idx], n) : 0.0;
		for (uint k = 0; k < pc.K; ++k)
			sum += OaLoad(heap[pc.x_idx], m * pc.K + k) *
				OaLoad(heap[pc.w2_idx], n * pc.K + k);
	}
	OaStore(heap[pc.out_idx], linear, sum);
}
