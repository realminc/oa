// NCL MaxUnpool1d forward: the recorded argmax scatter adjoint of MaxPool1d.
// Each output element scans the windows that cover it; the window whose index
// names this element contributes its pooled value. Overlapping windows resolve
// to the highest pooled index, matching the donor sequential scatter order.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint indices_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint output_length;
	uint input_length;
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_unpool_1d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint batch_channel = thread_id.y;
	if (
		output_x >= push.output_length
		|| batch_channel >= push.batch_size * push.channels
	) {
		return;
	}

	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint output_offset =
		batch * push.channels * push.output_length
		+ channel * push.output_length
		+ output_x;
	float value = 0.0f;
	for (uint input_index = 0; input_index < push.input_length; ++input_index) {
		int output_x_start = int(input_index * push.stride) - int(push.padding);
		int output_x_end = output_x_start + int(push.kernel_size);
		if (int(output_x) >= output_x_start && int(output_x) < output_x_end) {
			uint input_offset =
				batch * push.channels * push.input_length
				+ channel * push.input_length
				+ input_index;
			if (
				load_u32(storage_buffers[push.indices_index], input_offset)
				== output_offset
			) {
				value = load_f32(storage_buffers[push.input_index], input_offset);
			}
		}
	}
	store_f32(storage_buffers[push.output_index], output_offset, value);
}
