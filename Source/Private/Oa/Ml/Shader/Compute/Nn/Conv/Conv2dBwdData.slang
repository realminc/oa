// Conv2dBwdData — Backward pass for 2D convolution (input gradient)
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
	uint H;
	uint W;
	uint K;
	uint S;
	uint P;
	uint OutH;
	uint OutW;
	uint Groups;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.N * pc.InC * pc.H * pc.W;
	if (idx >= total) return;

	uint w = idx % pc.W;
	uint tmp = idx / pc.W;
	uint h = tmp % pc.H;
	tmp = tmp / pc.H;
	uint ic = tmp % pc.InC;
	uint n = tmp / pc.InC;

	float grad_sum = 0.0;

	uint inChannelsPerGroup = pc.InC / pc.Groups;
	uint outChannelsPerGroup = pc.OutC / pc.Groups;
	uint group = ic / inChannelsPerGroup;
	uint localIc = ic - group * inChannelsPerGroup;
	uint outputChannelBase = group * outChannelsPerGroup;

	for (uint localOc = 0; localOc < outChannelsPerGroup; ++localOc) {
		uint oc = outputChannelBase + localOc;
		for (uint kh = 0; kh < pc.K; ++kh) {
			for (uint kw = 0; kw < pc.K; ++kw) {
				int oh = int(h - kh + pc.P) / int(pc.S);
				int ow = int(w - kw + pc.P) / int(pc.S);

				if (oh >= 0 && oh < int(pc.OutH) && ow >= 0 && ow < int(pc.OutW) &&
					((h - kh + pc.P) % pc.S == 0) && ((w - kw + pc.P) % pc.S == 0)) {
					uint out_off = ((n * pc.OutC + oc) * pc.OutH + uint(oh)) * pc.OutW + uint(ow);
					uint w_off = ((oc * inChannelsPerGroup + localIc) * pc.K + kh) * pc.K + kw;
					grad_sum += OaLoad(heap[pc.d_out_idx], out_off) * OaLoad(heap[pc.weight_idx], w_off);
				}
			}
		}
	}

	uint in_off = ((n * pc.InC + ic) * pc.H + h) * pc.W + w;
	OaStore(heap[pc.d_input_idx], in_off, grad_sum);
}
