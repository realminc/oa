import Storage;

struct PushConstants {
	uint input_idx;
	uint weight_idx;
	uint bias_idx;
	uint out_idx;
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
	uint total = pc.N * pc.OutC * pc.OutH * pc.OutW;
	if (idx >= total) return;

	uint ow = idx % pc.OutW;
	uint tmp = idx / pc.OutW;
	uint oh = tmp % pc.OutH;
	tmp = tmp / pc.OutH;
	uint oc = tmp % pc.OutC;
	uint n = tmp / pc.OutC;

	float val = OaLoad(heap[pc.bias_idx], oc);

	uint inChannelsPerGroup = pc.InC / pc.Groups;
	uint outChannelsPerGroup = pc.OutC / pc.Groups;
	uint group = oc / outChannelsPerGroup;
	uint inputChannelBase = group * inChannelsPerGroup;

	for (uint localIc = 0; localIc < inChannelsPerGroup; ++localIc) {
		uint ic = inputChannelBase + localIc;
		for (uint kh = 0; kh < pc.K; ++kh) {
			for (uint kw = 0; kw < pc.K; ++kw) {
				int ih = int(oh * pc.S + kh) - int(pc.P);
				int iw = int(ow * pc.S + kw) - int(pc.P);
				if (ih >= 0 && ih < int(pc.H) && iw >= 0 && iw < int(pc.W)) {
					uint in_off = ((n * pc.InC + ic) * pc.H + uint(ih)) * pc.W + uint(iw);
					uint w_off = ((oc * inChannelsPerGroup + localIc) * pc.K + kh) * pc.K + kw;
					val += OaLoad(heap[pc.input_idx], in_off) * OaLoad(heap[pc.weight_idx], w_off);
				}
			}
		}
	}

	uint out_off = ((n * pc.OutC + oc) * pc.OutH + oh) * pc.OutW + ow;
	OaStore(heap[pc.out_idx], out_off, val);
}
