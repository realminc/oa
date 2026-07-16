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
	uint w = pc.w0_idx, b = pc.b0_idx, n = packedN;
	if (packedN >= pc.N0 + pc.N1) { w = pc.w2_idx; b = pc.b2_idx; n -= pc.N0 + pc.N1; }
	else if (packedN >= pc.N0) { w = pc.w1_idx; b = pc.b1_idx; n -= pc.N0; }
	float sum = pc.HasBias != 0 ? OaLoad(heap[b], n) : 0.0;
	for (uint k = 0; k < pc.K; ++k)
		sum += OaLoad(heap[pc.x_idx], m * pc.K + k) * OaLoad(heap[w], n * pc.K + k);
	OaStore(heap[pc.out_idx], linear, sum);
}
