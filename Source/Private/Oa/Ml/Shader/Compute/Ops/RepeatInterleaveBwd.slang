// RepeatInterleaveBwd — backward of RepeatInterleave.
// Gradient flows back by summing the InRepeats copies for each input element.
// Each thread accumulates one input element from InRepeats output gradient slots.

import Storage;

struct PushConstants {
	uint gradOut_idx;  // upstream gradient [output shape]
	uint gradIn_idx;   // output gradient [input shape]
	uint count;        // total INPUT elements
	uint repeats;      // repeat factor
	uint dim;          // dimension that was repeated
	uint rank;         // tensor rank
	uint outShape[4];  // output shape (up to 4D)
	uint inStride[4];  // input strides (row-major)
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	// Decompose input flat index to multi-index
	uint rem = idx;
	uint inCoords[4];
	for (int d = int(pc.rank) - 1; d >= 0; --d) {
		inCoords[d] = rem % pc.outShape[d];
		rem /= pc.outShape[d];
	}

	// Sum over the InRepeats copies in the output
	float sum = 0.0;
	for (uint r = 0; r < pc.repeats; ++r) {
		uint outFlat = 0;
		for (int d2 = 0; d2 < int(pc.rank); ++d2) {
			uint coord = inCoords[d2];
			if (d2 == int(pc.dim)) {
				coord = inCoords[d2] * pc.repeats + r;
			}
			// Row-major offset for this dimension
			uint stride = 1;
			for (int d3 = d2 + 1; d3 < int(pc.rank); ++d3) {
				stride *= pc.outShape[d3];
			}
			outFlat += coord * stride;
		}
		sum += OaLoad(heap[pc.gradOut_idx], outFlat);
	}

	OaStore(heap[pc.gradIn_idx], idx, sum);
}
