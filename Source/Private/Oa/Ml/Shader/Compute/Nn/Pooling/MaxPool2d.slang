// MaxPool2d Forward — 2D max pooling with argmax tracking
// Outputs: max values + argmax indices (flat indices into input)

import Storage;

struct PushConstants {
	uint in_idx;
	uint out_idx;
	uint indices_idx;
	uint BatchSize;
	uint Channels;
	uint InHeight;
	uint InWidth;
	uint OutHeight;
	uint OutWidth;
	uint KernelSize;
	uint Stride;
	uint Padding;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint out_h = tid.x;
	uint out_w = tid.y;
	uint batch_channel = tid.z;
	
	if (out_h >= pc.OutHeight || out_w >= pc.OutWidth || batch_channel >= pc.BatchSize * pc.Channels) {
		return;
	}
	
	uint batch = batch_channel / pc.Channels;
	uint channel = batch_channel % pc.Channels;
	
	// Calculate input window start (with padding)
	int in_h_start = int(out_h * pc.Stride) - int(pc.Padding);
	int in_w_start = int(out_w * pc.Stride) - int(pc.Padding);
	
	// Find max value and its position in the pooling window
	float max_val = -3.402823466e+38f; // -FLT_MAX
	uint max_idx = 0;
	
	for (uint kh = 0; kh < pc.KernelSize; ++kh) {
		for (uint kw = 0; kw < pc.KernelSize; ++kw) {
			int in_h = in_h_start + int(kh);
			int in_w = in_w_start + int(kw);
			
			// Check bounds (handle padding)
			if (in_h >= 0 && in_h < int(pc.InHeight) && in_w >= 0 && in_w < int(pc.InWidth)) {
				uint in_idx = batch * (pc.Channels * pc.InHeight * pc.InWidth)
				            + channel * (pc.InHeight * pc.InWidth)
				            + uint(in_h) * pc.InWidth
				            + uint(in_w);
				
				float val = OaLoad(heap[pc.in_idx], in_idx);
				
				if (val > max_val) {
					max_val = val;
					max_idx = in_idx;
				}
			}
		}
	}
	
	// Write output
	uint out_idx = batch * (pc.Channels * pc.OutHeight * pc.OutWidth)
	             + channel * (pc.OutHeight * pc.OutWidth)
	             + out_h * pc.OutWidth
	             + out_w;
	
	OaStore(heap[pc.out_idx], out_idx, max_val);
	OaStore(heap[pc.indices_idx], out_idx, float(max_idx)); // Store as float for compatibility
}