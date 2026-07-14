// AvgPool2d forward pass
// Input: [N, C, H, W] tensor
// Output: [N, C, H_out, W_out] tensor where each output element is the average of a kernel_size × kernel_size window
// Push constants: buffer indices (auto-prepended) + kernel_size, stride, padding, input dims, output dims

import Storage;

struct PushConstants {
	uint in_idx;
	uint out_idx;
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
	uint out_y = tid.x;
	uint out_x = tid.y;
	uint c_n = tid.z;  // combined channel and batch index
	
	if (out_y >= pc.out_height || out_x >= pc.out_width || c_n >= pc.batch_size * pc.channels) return;
	
	uint n = c_n / pc.channels;
	uint c = c_n % pc.channels;
	
	// Calculate input window bounds
	int in_y_start = int(out_y * pc.stride) - int(pc.padding);
	int in_x_start = int(out_x * pc.stride) - int(pc.padding);
	
	float sum = 0.0f;
	uint count = 0;
	
	// Average over kernel window
	for (uint ky = 0; ky < pc.kernel_size; ++ky) {
		for (uint kx = 0; kx < pc.kernel_size; ++kx) {
			int in_y = in_y_start + int(ky);
			int in_x = in_x_start + int(kx);
			
			// Check bounds (zero padding)
			if (in_y >= 0 && in_y < int(pc.in_height) && in_x >= 0 && in_x < int(pc.in_width)) {
				uint in_idx = n * pc.channels * pc.in_height * pc.in_width +
				              c * pc.in_height * pc.in_width +
				              uint(in_y) * pc.in_width +
				              uint(in_x);
				sum += OaLoad(heap[pc.in_idx], in_idx);
				count++;
			}
		}
	}
	
	// Write average
	uint out_idx = n * pc.channels * pc.out_height * pc.out_width +
	               c * pc.out_height * pc.out_width +
	               out_y * pc.out_width +
	               out_x;
	
	float avg = (count > 0) ? (sum / float(count)) : 0.0f;
	OaStore(heap[pc.out_idx], out_idx, avg);
}