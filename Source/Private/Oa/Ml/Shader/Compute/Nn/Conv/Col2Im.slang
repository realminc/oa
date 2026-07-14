// Col2Im: Transform columns back into image (inverse of Im2Col)
// Input: [N, C*K*K, OutH*OutW]
// Output: [N, C, H, W]
// Accumulates overlapping patches

import Storage;

struct PushConstants {
	uint input_idx;    // [N, C*K*K, OutH*OutW]
	uint output_idx;   // [N, C, H, W]
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
	uint total = pc.N * pc.C * pc.H * pc.W;
	if (idx >= total) return;
	
	// Decode: idx = n*(C*H*W) + c*(H*W) + h*W + w
	uint n = idx / (pc.C * pc.H * pc.W);
	uint rem = idx % (pc.C * pc.H * pc.W);
	uint c = rem / (pc.H * pc.W);
	rem = rem % (pc.H * pc.W);
	uint h = rem / pc.W;
	uint w = rem % pc.W;
	
	float sum = 0.0f;
	
	// Accumulate from all patches that overlap this position
	for (uint kh = 0; kh < pc.KH; ++kh) {
		for (uint kw = 0; kw < pc.KW; ++kw) {
			// Check if this kernel position contributes to (h, w)
			// h = oh * stride + kh - padding
			// oh = (h + padding - kh) / stride
			int oh_times_stride = (int)h + (int)pc.padding - (int)kh;
			if (oh_times_stride < 0 || oh_times_stride % (int)pc.stride != 0) continue;
			
			uint oh = (uint)(oh_times_stride / (int)pc.stride);
			if (oh >= pc.OutH) continue;
			
			int ow_times_stride = (int)w + (int)pc.padding - (int)kw;
			if (ow_times_stride < 0 || ow_times_stride % (int)pc.stride != 0) continue;
			
			uint ow = (uint)(ow_times_stride / (int)pc.stride);
			if (ow >= pc.OutW) continue;
			
			// Read from column matrix
			uint col_idx = c * pc.KH * pc.KW + kh * pc.KW + kw;
			uint spatial_idx = oh * pc.OutW + ow;
			uint input_offset = n * (pc.C * pc.KH * pc.KW) * (pc.OutH * pc.OutW) + 
			                    col_idx * (pc.OutH * pc.OutW) + spatial_idx;
			
			sum += OaLoad(heap[pc.input_idx], input_offset);
		}
	}
	
	OaStore(heap[pc.output_idx], idx, sum);
}