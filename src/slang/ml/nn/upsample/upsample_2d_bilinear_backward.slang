// Deterministic gather-style bilinear upsampling adjoint for FP32 NCHW matrices.

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_gradient_index;
	uint batch_size;
	uint channels;
	uint input_height;
	uint input_width;
	uint output_height;
	uint output_width;
	uint scale_factor;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("upsample_2d_bilinear_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint input_x = group_id.x * 16u + group_thread_id.x;
	uint input_y = group_id.y * 16u + group_thread_id.y;
	uint batch_channel = group_id.z;
	if (input_x >= push.input_width || input_y >= push.input_height) {
		return;
	}
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	if (batch >= push.batch_size) {
		return;
	}

	float gradient = 0.0f;
	for (uint output_y = 0u; output_y < push.output_height; ++output_y) {
		float source_y = (float(output_y) + 0.5f) / float(push.scale_factor) - 0.5f;
		int raw_y0 = int(floor(source_y));
		int raw_y1 = raw_y0 + 1;
		float y_fraction = source_y - float(raw_y0);
		uint y0 = uint(clamp(raw_y0, 0, int(push.input_height) - 1));
		uint y1 = uint(clamp(raw_y1, 0, int(push.input_height) - 1));
		for (uint output_x = 0u; output_x < push.output_width; ++output_x) {
			float source_x = (float(output_x) + 0.5f) / float(push.scale_factor) - 0.5f;
			int raw_x0 = int(floor(source_x));
			int raw_x1 = raw_x0 + 1;
			float x_fraction = source_x - float(raw_x0);
			uint x0 = uint(clamp(raw_x0, 0, int(push.input_width) - 1));
			uint x1 = uint(clamp(raw_x1, 0, int(push.input_width) - 1));
			uint output_index = ((batch * push.channels + channel) * push.output_height + output_y)
				* push.output_width + output_x;
			float output_gradient = load_f32(
				storage_buffers[push.output_gradient_index], output_index
			);
			if (input_y == y0 && input_x == x0) {
				gradient += output_gradient * (1.0f - y_fraction) * (1.0f - x_fraction);
			}
			if (input_y == y0 && input_x == x1) {
				gradient += output_gradient * (1.0f - y_fraction) * x_fraction;
			}
			if (input_y == y1 && input_x == x0) {
				gradient += output_gradient * y_fraction * (1.0f - x_fraction);
			}
			if (input_y == y1 && input_x == x1) {
				gradient += output_gradient * y_fraction * x_fraction;
			}
		}
	}
	uint input_index = ((batch * push.channels + channel) * push.input_height + input_y)
		* push.input_width + input_x;
	store_f32(storage_buffers[push.input_gradient_index], input_index, gradient);
}
