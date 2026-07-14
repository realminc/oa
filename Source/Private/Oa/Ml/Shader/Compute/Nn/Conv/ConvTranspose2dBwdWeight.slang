// ConvTranspose2dBwdWeight — Backward pass for 2D transposed convolution.
// Computes: d_weight = input conv_transpose(d_output), d_bias = sum(d_output)
//
// Input layout:  x       [N, InC, H, W]
//                d_out   [N, OutC, OutH, OutW]
//                d_weight [InC, OutC, K, K]
//                d_bias  [OutC]

import Storage;

struct PushConstants {
	uint input_idx;
	uint d_out_idx;
	uint d_weight_idx;
	uint d_bias_idx;
	uint N;
	uint InC;
	uint OutC;
	uint H;
	uint W;
	uint K;
	uint S;
	uint P;
	uint OutH;
	uint OutW;
	uint Total;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.Total) return;

	uint weightCount = pc.InC * pc.OutC * pc.K * pc.K;

	// Compute weight gradient.
	if (idx < weightCount) {
		uint kw = idx % pc.K;
		uint tmp = idx / pc.K;
		uint kh = tmp % pc.K;
		tmp = tmp / pc.K;
		uint oc = tmp % pc.OutC;
		uint ic = tmp / pc.OutC;

		float grad_sum = 0.0f;

		for (uint n = 0; n < pc.N; ++n) {
			for (uint oh = 0; oh < pc.OutH; ++oh) {
				int ih_raw = int(oh + pc.P) - int(kh);
				if (ih_raw < 0 || (uint(ih_raw) % pc.S) != 0) continue;
				int ih = ih_raw / int(pc.S);
				if (ih >= int(pc.H)) continue;
				for (uint ow = 0; ow < pc.OutW; ++ow) {
					int iw_raw = int(ow + pc.P) - int(kw);
					if (iw_raw < 0 || (uint(iw_raw) % pc.S) != 0) continue;
					int iw = iw_raw / int(pc.S);
					if (iw >= int(pc.W)) continue;

					uint in_off = ((n * pc.InC + ic) * pc.H + uint(ih)) * pc.W + uint(iw);
					uint out_off = ((n * pc.OutC + oc) * pc.OutH + oh) * pc.OutW + ow;
					grad_sum += OaLoad(heap[pc.input_idx], in_off) * OaLoad(heap[pc.d_out_idx], out_off);
				}
			}
		}

		uint w_off = ((ic * pc.OutC + oc) * pc.K + kh) * pc.K + kw;
		OaStore(heap[pc.d_weight_idx], w_off, grad_sum);
		return;
	}

	// Compute bias gradient.
	uint biasIdx = idx - weightCount;
	if (biasIdx >= pc.OutC) return;

	float bias_sum = 0.0f;
	for (uint n = 0; n < pc.N; ++n) {
		for (uint oh = 0; oh < pc.OutH; ++oh) {
			for (uint ow = 0; ow < pc.OutW; ++ow) {
				uint out_off = ((n * pc.OutC + biasIdx) * pc.OutH + oh) * pc.OutW + ow;
				bias_sum += OaLoad(heap[pc.d_out_idx], out_off);
			}
		}
	}
	OaStore(heap[pc.d_bias_idx], biasIdx, bias_sum);
}
