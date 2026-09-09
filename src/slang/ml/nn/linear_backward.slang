// FP32 input adjoint for one linear projection.

import storage;
import attributes;

struct PushConstants {
	uint weight_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint batch;
	uint input_features;
	uint output_features;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("linear_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint input_count = push.batch * push.input_features;
	if (index >= input_count) {
		return;
	}

	uint row = index / push.input_features;
	uint input_feature = index % push.input_features;
	float gradient = 0.0f;
	for (uint output_feature = 0; output_feature < push.output_features; ++output_feature) {
		gradient += load_f32(
			storage_buffers[push.output_gradient_index],
			row * push.output_features + output_feature
		) * load_f32(
			storage_buffers[push.weight_index],
			output_feature * push.input_features + input_feature
		);
	}
	store_f32(storage_buffers[push.input_gradient_index], index, gradient);
}
