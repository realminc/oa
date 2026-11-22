// Mechanical adaptation of OA's keep-dimension Sum reduction applied to two
// same-geometry inputs in one pass; used by norm adjoints that reduce a
// weight contribution and the raw output gradient over the same axis.

import storage;
import attributes;

struct PushConstants {
	uint input_a_index;
	uint input_b_index;
	uint output_a_index;
	uint output_b_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sum_axis_pair")]
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
	float accumulator_a = 0.0f;
	float accumulator_b = 0.0f;
	for (uint axis_index = 0; axis_index < push.dim_size; ++axis_index) {
		uint element = base + axis_index * push.inner_size;
		accumulator_a += load_f32(storage_buffers[push.input_a_index], element);
		accumulator_b += load_f32(storage_buffers[push.input_b_index], element);
	}
	store_f32(storage_buffers[push.output_a_index], output_index, accumulator_a);
	store_f32(storage_buffers[push.output_b_index], output_index, accumulator_b);
}
