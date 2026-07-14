// BiasAdd — Broadcast addition along last dimension (in-place)
// Stream.cpp prepends buffer indices (biased, bias) to push constants.
// Host passes: rows, cols

import Storage;

struct PushConstants {
	uint biased_idx;     // Stream.cpp prepends: bufs[0]
	uint bias_idx;       // Stream.cpp prepends: bufs[1]
	uint rows;           // Host parameter starts here
	uint cols;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// In-place: biased[row * cols + col] += bias[col]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.rows * pc.cols) return;
	uint col = idx % pc.cols;
	OaStore(heap[pc.biased_idx], idx,
		OaLoad(heap[pc.biased_idx], idx) + OaLoad(heap[pc.bias_idx], col));
}
