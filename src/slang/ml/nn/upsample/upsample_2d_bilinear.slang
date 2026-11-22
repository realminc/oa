// Half-pixel, align-corners-false bilinear upsampling for FP32 NCHW matrices.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
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

float input_value(uint batch, uint channel, uint y, uint x) {
	uint index = ((batch * push.channels + channel) * push.input_height + y)
		* push.input_width + x;
	return load_f32(storage_buffers[push.input_index], index);
}

[kernel_name("upsample_2d_bilinear")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint output_x = group_id.x * 16u + group_thread_id.x;
	uint output_y = group_id.y * 16u + group_thread_id.y;
	uint batch_channel = group_id.z;
	if (output_x >= push.output_width || output_y >= push.output_height) {
		return;
	}
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	if (batch >= push.batch_size) {
		return;
	}

	float input_y = (float(output_y) + 0.5f) / float(push.scale_factor) - 0.5f;
	float input_x = (float(output_x) + 0.5f) / float(push.scale_factor) - 0.5f;
	input_y = clamp(input_y, 0.0f, float(push.input_height - 1u));
	input_x = clamp(input_x, 0.0f, float(push.input_width - 1u));
	uint y0 = uint(floor(input_y));
	uint x0 = uint(floor(input_x));
	uint y1 = min(y0 + 1u, push.input_height - 1u);
	uint x1 = min(x0 + 1u, push.input_width - 1u);
	float y_fraction = input_y - float(y0);
	float x_fraction = input_x - float(x0);
	float top = lerp(
		input_value(batch, channel, y0, x0),
		input_value(batch, channel, y0, x1),
		x_fraction
	);
	float bottom = lerp(
		input_value(batch, channel, y1, x0),
		input_value(batch, channel, y1, x1),
		x_fraction
	);
	uint output_index = ((batch * push.channels + channel) * push.output_height + output_y)
		* push.output_width + output_x;
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		lerp(top, bottom, y_fraction)
	);
}
