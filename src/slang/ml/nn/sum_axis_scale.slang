// Axis-parallel FP32 sum reduction with a scalar-multiplied epilogue write.
// Fused lowering of `sum_axis -> scale`: stores the raw reduction and the
// scaled result so the intermediate sum remains observable.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint scaled_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sum_axis_scale")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint output_index = dispatch_thread_id.x;
	uint output_count = push.outer_size * push.inner_size;
	if (output_index >= output_count) {
		return;
	}

	uint inner = output_index % push.inner_size;
	uint outer = output_index / push.inner_size;
	uint base = outer * push.dim_size * push.inner_size + inner;
	float accumulator = 0.0f;
	for (uint axis_index = 0; axis_index < push.dim_size; ++axis_index) {
		accumulator += load_f32(
			storage_buffers[push.input_index],
			base + axis_index * push.inner_size
		);
	}
	store_f32(storage_buffers[push.output_index], output_index, accumulator);
	store_f32(storage_buffers[push.scaled_index], output_index, accumulator * push.scale);
}
