// Conv1dBwdData — Backward pass for 1D convolution (input gradient)
// Computes: d_input = d_output conv_transpose(weight)
// Transposed convolution to compute gradient with respect to input

import Storage;

struct PushConstants {
	uint d_out_idx;
	uint weight_idx;
	uint d_input_idx;
	uint N;
	uint InC;
	uint OutC;
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
	if (idx >= total) return;

	uint l = idx % pc.L;
	uint tmp = idx / pc.L;
	uint ic = tmp % pc.InC;
	uint n = tmp / pc.InC;

	float grad_sum = 0.0;

	for (uint oc = 0; oc < pc.OutC; ++oc) {
		for (uint k = 0; k < pc.K; ++k) {
			int offset = int(l) - int(k) * int(pc.D) + int(pc.P);
			int ol = offset / int(pc.S);

			if (ol >= 0 && ol < int(pc.OutL) && (offset % int(pc.S) == 0)) {
				uint out_off = (n * pc.OutC + oc) * pc.OutL + uint(ol);
				uint w_off = (oc * pc.InC + ic) * pc.K + k;
				grad_sum += OaLoad(heap[pc.d_out_idx], out_off) * OaLoad(heap[pc.weight_idx], w_off);
			}
		}
	}

	uint in_off = (n * pc.InC + ic) * pc.L + l;
	OaStore(heap[pc.d_input_idx], in_off, grad_sum);
}
