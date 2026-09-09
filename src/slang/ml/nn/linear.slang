// FP32 linear projection using OA's [output, input] weight layout.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint batch;
	uint input_features;
	uint output_features;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("linear")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint element_count = push.batch * push.output_features;
	if (index >= element_count) {
		return;
	}

	uint row = index / push.output_features;
	uint output_feature = index % push.output_features;
	float value = load_f32(storage_buffers[push.bias_index], output_feature);
	for (uint input_feature = 0; input_feature < push.input_features; ++input_feature) {
		value += load_f32(
			storage_buffers[push.input_index],
			row * push.input_features + input_feature
		) * load_f32(
			storage_buffers[push.weight_index],
			output_feature * push.input_features + input_feature
		);
	}
	store_f32(storage_buffers[push.output_index], index, value);
}
