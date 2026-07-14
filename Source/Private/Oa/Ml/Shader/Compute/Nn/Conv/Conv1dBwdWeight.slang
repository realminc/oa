// Conv1dBwdWeight — Backward pass for 1D convolution (weight gradient)
// Computes: d_weight = input conv_transpose(d_output)
// Fused weight and bias gradient computation

import Storage;

struct PushConstants {
	uint input_idx;
	uint d_out_idx;
	uint d_weight_idx;
	uint d_bias_idx;
	uint N;
	uint InC;
	uint OutC;
	uint L;
	uint K;
	uint S;
	uint P;
	uint D;
	uint OutL;
	uint Total;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.Total) return;

	uint weightCount = pc.OutC * pc.InC * pc.K;
	
	// Compute weight gradient
	if (idx < weightCount) {
		uint k = idx % pc.K;
		uint tmp = idx / pc.K;
		uint ic = tmp % pc.InC;
		uint oc = tmp / pc.InC;

		float grad_sum = 0.0;

		for (uint n = 0; n < pc.N; ++n) {
			for (uint ol = 0; ol < pc.OutL; ++ol) {
				int il = int(ol * pc.S + k * pc.D) - int(pc.P);

				if (il >= 0 && il < int(pc.L)) {
					uint in_off = (n * pc.InC + ic) * pc.L + uint(il);
					uint out_off = (n * pc.OutC + oc) * pc.OutL + ol;
					grad_sum += OaLoad(heap[pc.input_idx], in_off) * OaLoad(heap[pc.d_out_idx], out_off);
				}
			}
		}

		uint w_off = (oc * pc.InC + ic) * pc.K + k;
		OaStore(heap[pc.d_weight_idx], w_off, grad_sum);
		return;
	}

	// Compute bias gradient
	uint biasIdx = idx - weightCount;
	if (biasIdx >= pc.OutC) return;

	float bias_sum = 0.0;
	for (uint n = 0; n < pc.N; ++n) {
		for (uint ol = 0; ol < pc.OutL; ++ol) {
			uint out_off = (n * pc.OutC + biasIdx) * pc.OutL + ol;
			bias_sum += OaLoad(heap[pc.d_out_idx], out_off);
		}
	}
	OaStore(heap[pc.d_bias_idx], biasIdx, bias_sum);
}
