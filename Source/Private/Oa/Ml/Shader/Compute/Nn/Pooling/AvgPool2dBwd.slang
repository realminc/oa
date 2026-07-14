// AvgPool2d backward pass
// Distributes output gradient evenly across the pooling window
// Input: grad_output [N, C, H_out, W_out]
// Output: grad_input [N, C, H_in, W_in]
// Each output gradient is divided by window size and added to all input positions in the window

import Storage;

struct PushConstants {
	uint grad_out_idx;
	uint grad_in_idx;
	uint batch_size;
	uint channels;
	uint in_height;
	uint in_width;
	uint out_height;
	uint out_width;
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint in_y = tid.x;
	uint in_x = tid.y;
	uint c_n = tid.z;  // combined channel and batch index
	
	if (in_y >= pc.in_height || in_x >= pc.in_width || c_n >= pc.batch_size * pc.channels) return;
	
	uint n = c_n / pc.channels;
	uint c = c_n % pc.channels;
	
	float grad_sum = 0.0f;
	
	// Find all output positions that this input contributes to
	// For each output position, check if this input is in its pooling window
	for (uint out_y = 0; out_y < pc.out_height; ++out_y) {
		for (uint out_x = 0; out_x < pc.out_width; ++out_x) {
			// Calculate input window bounds for this output position
			int in_y_start = int(out_y * pc.stride) - int(pc.padding);
			int in_x_start = int(out_x * pc.stride) - int(pc.padding);
			int in_y_end = in_y_start + int(pc.kernel_size);
			int in_x_end = in_x_start + int(pc.kernel_size);
			
			// Check if current input position is in this window
			if (int(in_y) >= in_y_start && int(in_y) < in_y_end &&
			    int(in_x) >= in_x_start && int(in_x) < in_x_end) {
				
				// Count valid positions in this window (for average)
				uint count = 0;
				for (uint ky = 0; ky < pc.kernel_size; ++ky) {
					for (uint kx = 0; kx < pc.kernel_size; ++kx) {
						int y = in_y_start + int(ky);
						int x = in_x_start + int(kx);
						if (y >= 0 && y < int(pc.in_height) && x >= 0 && x < int(pc.in_width)) {
							count++;
						}
					}
				}
				
				// Get output gradient and distribute
				uint out_idx = n * pc.channels * pc.out_height * pc.out_width +
				               c * pc.out_height * pc.out_width +
				               out_y * pc.out_width +
				               out_x;
				
				float grad_out = OaLoad(heap[pc.grad_out_idx], out_idx);
				grad_sum += grad_out / float(count);
			}
		}
	}
	
	// Write accumulated gradient
	uint in_idx = n * pc.channels * pc.in_height * pc.in_width +
	              c * pc.in_height * pc.in_width +
	              in_y * pc.in_width +
	              in_x;
	
	OaStore(heap[pc.grad_in_idx], in_idx, grad_sum);
}