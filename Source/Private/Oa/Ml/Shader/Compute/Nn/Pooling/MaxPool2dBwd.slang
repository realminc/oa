// MaxPool2dBwd — Backward pass for 2D max pooling
// Scatters gradient to the positions that were selected as max

import Storage;

struct PushConstants {
	uint grad_in_idx;
	uint grad_out_idx;
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
	uint in_h = tid.x;
	uint in_w = tid.y;
	uint batch_channel = tid.z;
	
	if (in_h >= pc.InHeight || in_w >= pc.InWidth || batch_channel >= pc.BatchSize * pc.Channels) {
		return;
	}
	
	uint batch = batch_channel / pc.Channels;
	uint channel = batch_channel % pc.Channels;
	
	uint in_idx = batch * (pc.Channels * pc.InHeight * pc.InWidth)
	            + channel * (pc.InHeight * pc.InWidth)
	            + in_h * pc.InWidth
	            + in_w;
	
	float grad_sum = 0.0f;
	
	// Check all output positions that could have selected this input position
	// For each output position, check if its argmax index matches this input position
	for (uint out_h = 0; out_h < pc.OutHeight; ++out_h) {
		for (uint out_w = 0; out_w < pc.OutWidth; ++out_w) {
			// Calculate input window for this output position
			int in_h_start = int(out_h * pc.Stride) - int(pc.Padding);
			int in_w_start = int(out_w * pc.Stride) - int(pc.Padding);
			int in_h_end = in_h_start + int(pc.KernelSize);
			int in_w_end = in_w_start + int(pc.KernelSize);
			
			// Check if current input position is within this window
			if (int(in_h) >= in_h_start && int(in_h) < in_h_end &&
			    int(in_w) >= in_w_start && int(in_w) < in_w_end) {
				
				uint out_idx = batch * (pc.Channels * pc.OutHeight * pc.OutWidth)
				             + channel * (pc.OutHeight * pc.OutWidth)
				             + out_h * pc.OutWidth
				             + out_w;
				
				// Check if this input position was the argmax for this output
				uint max_idx = uint(OaLoad(heap[pc.indices_idx], out_idx));
				
				if (max_idx == in_idx) {
					// This input was selected as max - accumulate gradient
					float grad_out = OaLoad(heap[pc.grad_out_idx], out_idx);
					grad_sum += grad_out;
				}
			}
		}
	}
	
	// Write accumulated gradient
	OaStore(heap[pc.grad_in_idx], in_idx, grad_sum);
}