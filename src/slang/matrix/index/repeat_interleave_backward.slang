// Donor: source/cpp/lib/oa/ml/shader/compute/ops/repeatInterleaveBwd.slang
// Sum repeated FP32 output adjoints into one rank-1..4 input adjoint.

import attributes;
import storage;

struct PushConstants {
	uint gradient_output_index;
	uint gradient_input_index;
	uint count;
	uint repeats;
	uint dim;
	uint rank;
	uint input_dim_0;
	uint input_dim_1;
	uint input_dim_2;
	uint input_dim_3;
	uint output_stride_0;
	uint output_stride_1;
	uint output_stride_2;
	uint output_stride_3;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("repeat_interleave_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) {
		return;
	}
	uint input_dims[4] = {
		push.input_dim_0, push.input_dim_1, push.input_dim_2, push.input_dim_3
	};
	uint output_strides[4] = {
		push.output_stride_0, push.output_stride_1,
		push.output_stride_2, push.output_stride_3
	};
	uint coordinates[4] = { 0, 0, 0, 0 };
	uint remainder = index;
	for (int axis = int(push.rank) - 1; axis >= 0; --axis) {
		coordinates[axis] = remainder % input_dims[axis];
		remainder /= input_dims[axis];
	}
	float sum = 0.0;
	for (uint repeat = 0; repeat < push.repeats; ++repeat) {
		uint output_offset = 0;
		for (uint axis = 0; axis < push.rank; ++axis) {
			uint coordinate = coordinates[axis];
			if (axis == push.dim) {
				coordinate = coordinate * push.repeats + repeat;
			}
			output_offset += coordinate * output_strides[axis];
		}
		sum += load_f32(storage_buffers[push.gradient_output_index], output_offset);
	}
	store_f32(storage_buffers[push.gradient_input_index], index, sum);
}
