// Column-wise sum: out[c] = sum_r input[r * cols + c]
// Reduces a [rows, cols] matrix along rows to produce [cols].
// One thread per output column.

import Storage;

struct PushConstants {
	uint input_idx;
	uint out_idx;
	uint rows;
	uint cols;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint col = tid.x;
	if (col >= pc.cols) return;

	float sum = 0.0;
	for (uint r = 0; r < pc.rows; r++) {
		sum += OaLoad(heap[pc.input_idx], r * pc.cols + col);
	}
	OaStore(heap[pc.out_idx], col, sum);
}
