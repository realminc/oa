// Im2Col1d: unfold a 1-D conv input into a GEMM-ready column matrix so the
// convolution can run as a single tensor-core matmul instead of a scalar
// direct conv. Output layout is chosen so `Linear(cols, wReshaped, bias)`
// reproduces the convolution exactly:
//
//   input   x    : [N, InC, L]
//   output  cols : [N*OutL, InC*K]   (row = n*OutL + ol, col = ic*K + k)
//                  cols[row, col] = x[n, ic, ol*S + k*D - P]  (0 if OOB)
//
// One thread per output element. See FnMatrixConv1dGemm.cpp for the wrapper.

import Storage;

struct PushConstants {
	uint x_idx;    // [N, InC, L]
	uint out_idx;  // [N*OutL, InC*K]
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
	uint innerK = pc.InC * pc.K;             // columns of the GEMM (K dimension)
	uint total = pc.N * pc.OutL * innerK;
	if (idx >= total)
		return;

	uint col = idx % innerK;                 // ic*K + k
	uint row = idx / innerK;                 // n*OutL + ol
	uint k = col % pc.K;
	uint ic = col / pc.K;
	uint ol = row % pc.OutL;
	uint n = row / pc.OutL;

	int inPos = int(ol) * int(pc.S) + int(k) * int(pc.D) - int(pc.P);
	float val = 0.0;
	if (inPos >= 0 && uint(inPos) < pc.L) {
		uint xLin = (n * pc.InC + ic) * pc.L + uint(inPos);
		val = OaLoad(heap[pc.x_idx], xLin);
	}
	// Row-major [N*OutL, InC*K]: linear index == idx by construction.
	OaStore(heap[pc.out_idx], idx, val);
}
