// Col2Im1d: backward of Im2Col1d — fold the column-matrix gradient back into
// the input-shaped gradient, accumulating over every receptive field that read
// each input element. Race-free: one thread per input element gathers all the
// columns that touched it (no atomics).
//
//   dcols : [N*OutL, InC*K]   (row = n*OutL + ol, col = ic*K + k)
//   dx    : [N, InC, L]
//   dx[n, ic, l] = sum over (ol, k) with ol*S + k*D - P == l of dcols[row, col]

import Storage;

struct PushConstants {
	uint dcols_idx;  // [N*OutL, InC*K]
	uint dx_idx;     // [N, InC, L]
	uint N;
	uint InC;
	uint L;
	uint K;
	uint S;
	uint P;
	uint D;
	uint OutL;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.N * pc.InC * pc.L;
	if (idx >= total)
		return;

	uint l = idx % pc.L;
	uint tmp = idx / pc.L;
	uint ic = tmp % pc.InC;
	uint n = tmp / pc.InC;

	uint innerK = pc.InC * pc.K;
	float sum = 0.0;
	// For each kernel tap k, find the (single) output position ol that read
	// input index l through this tap: ol*S + k*D - P == l.
	for (uint k = 0; k < pc.K; k++) {
		int num = int(l) + int(pc.P) - int(k) * int(pc.D);
		if (num < 0)
			continue;
		if ((uint(num) % pc.S) != 0)
			continue;
		uint ol = uint(num) / pc.S;
		if (ol >= pc.OutL)
			continue;
		uint row = n * pc.OutL + ol;
		uint col = ic * pc.K + k;
		sum += OaLoad(heap[pc.dcols_idx], row * innerK + col);
	}
	OaStore(heap[pc.dx_idx], idx, sum);
}
