// RepeatInterleave — repeats each element along a dimension InRepeats times.
// Each thread copies one output element from the corresponding input element.
// Output shape = input shape with dim[InDim] *= InRepeats.

import Storage;

struct PushConstants {
	uint x_idx;    // input
	uint out_idx;  // output
	uint count;    // total output elements
	uint repeats;  // repeat factor
	uint dim;      // dimension to repeat along
	uint rank;     // tensor rank
	uint inShape[4];   // input shape (up to 4D)
	uint inStride[4];  // input strides (row-major)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	// Compute output multi-index, then map to input index
	uint rem = idx;
	uint srcFlat = 0;
	for (int d = int(pc.rank) - 1; d >= 0; --d) {
		uint coord = rem % (pc.inShape[d] * ((d == int(pc.dim)) ? pc.repeats : 1u));
		rem /= (pc.inShape[d] * ((d == int(pc.dim)) ? pc.repeats : 1u));
		uint srcCoord = (d == int(pc.dim)) ? (coord / pc.repeats) : coord;
		srcFlat += srcCoord * pc.inStride[d];
	}

	OaStore(heap[pc.out_idx], idx, OaLoad(heap[pc.x_idx], srcFlat));
}
