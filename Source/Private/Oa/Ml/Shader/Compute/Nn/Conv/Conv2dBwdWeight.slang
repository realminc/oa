// Conv2dBwdWeight — Backward pass for 2D convolution (weight gradient)
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
	uint H;
	uint W;
	uint K;
	uint S;
	uint P;
	uint OutH;
	uint OutW;
	uint Total;
	uint Groups;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.Total) return;

	uint inChannelsPerGroup = pc.InC / pc.Groups;
	uint outChannelsPerGroup = pc.OutC / pc.Groups;
	uint weightCount = pc.OutC * inChannelsPerGroup * pc.K * pc.K;
	
	// Compute weight gradient
	if (idx < weightCount) {
		uint kw = idx % pc.K;
		uint tmp = idx / pc.K;
		uint kh = tmp % pc.K;
		tmp = tmp / pc.K;
		uint localIc = tmp % inChannelsPerGroup;
		uint oc = tmp / inChannelsPerGroup;
		uint group = oc / outChannelsPerGroup;
		uint ic = group * inChannelsPerGroup + localIc;

		float grad_sum = 0.0;

		for (uint n = 0; n < pc.N; ++n) {
			for (uint oh = 0; oh < pc.OutH; ++oh) {
				for (uint ow = 0; ow < pc.OutW; ++ow) {
					int ih = int(oh * pc.S + kh) - int(pc.P);
					int iw = int(ow * pc.S + kw) - int(pc.P);

					if (ih >= 0 && ih < int(pc.H) && iw >= 0 && iw < int(pc.W)) {
						uint in_off = ((n * pc.InC + ic) * pc.H + uint(ih)) * pc.W + uint(iw);
						uint out_off = ((n * pc.OutC + oc) * pc.OutH + oh) * pc.OutW + ow;
						grad_sum += OaLoad(heap[pc.input_idx], in_off) * OaLoad(heap[pc.d_out_idx], out_off);
					}
				}
			}
		}

		uint w_off = ((oc * inChannelsPerGroup + localIc) * pc.K + kh) * pc.K + kw;
		OaStore(heap[pc.d_weight_idx], w_off, grad_sum);
		return;
	}

	// Compute bias gradient
	uint biasIdx = idx - weightCount;
	if (biasIdx >= pc.OutC) return;

	float bias_sum = 0.0;
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
