// Im2Col: Transform image patches into columns for convolution via matrix multiplication
// Input: [N, C, H, W]
// Output: [N, C*K*K, OutH*OutW] where K is kernel size

import Storage;

struct PushConstants {
	uint input_idx;    // [N, C, H, W]
	uint output_idx;   // [N, C*K*K, OutH*OutW]
	uint N;
	uint C;
	uint H;
	uint W;
	uint OutH;
	uint OutW;
	uint KH;
	uint KW;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	uint total = pc.N * pc.C * pc.KH * pc.KW * pc.OutH * pc.OutW;
	if (idx >= total) return;
	
	// Decode: idx = n*(C*KH*KW*OutH*OutW) + c*(KH*KW*OutH*OutW) + kh*(KW*OutH*OutW) + kw*(OutH*OutW) + oh*OutW + ow
	uint n = idx / (pc.C * pc.KH * pc.KW * pc.OutH * pc.OutW);
	uint rem = idx % (pc.C * pc.KH * pc.KW * pc.OutH * pc.OutW);
	
	uint c = rem / (pc.KH * pc.KW * pc.OutH * pc.OutW);
	rem = rem % (pc.KH * pc.KW * pc.OutH * pc.OutW);
	
	uint kh = rem / (pc.KW * pc.OutH * pc.OutW);
	rem = rem % (pc.KW * pc.OutH * pc.OutW);
	
	uint kw = rem / (pc.OutH * pc.OutW);
	rem = rem % (pc.OutH * pc.OutW);
	
	uint oh = rem / pc.OutW;
	uint ow = rem % pc.OutW;
	
	// Compute input position
	int h = (int)(oh * pc.stride + kh) - (int)pc.padding;
	int w = (int)(ow * pc.stride + kw) - (int)pc.padding;
	
	float val = 0.0f;
	if (h >= 0 && h < (int)pc.H && w >= 0 && w < (int)pc.W) {
		uint input_offset = n * pc.C * pc.H * pc.W + c * pc.H * pc.W + (uint)h * pc.W + (uint)w;
		val = OaLoad(heap[pc.input_idx], input_offset);
	}
	
	// Output layout: [N, C*KH*KW, OutH*OutW]
	uint col_idx = c * pc.KH * pc.KW + kh * pc.KW + kw;
	uint spatial_idx = oh * pc.OutW + ow;
	uint output_offset = n * (pc.C * pc.KH * pc.KW) * (pc.OutH * pc.OutW) + col_idx * (pc.OutH * pc.OutW) + spatial_idx;
	
	OaStore(heap[pc.output_idx], output_offset, val);
}