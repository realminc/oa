// Donor: source/cpp/lib/oa/ml/shader/compute/ops/repeatInterleave.slang
// Repeat every FP32 element along one rank-1..4 dimension.

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint repeats;
	uint dim;
	uint rank;
	uint input_dim_0;
	uint input_dim_1;
	uint input_dim_2;
	uint input_dim_3;
	uint input_stride_0;
	uint input_stride_1;
	uint input_stride_2;
	uint input_stride_3;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("repeat_interleave")]
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
	uint input_strides[4] = {
		push.input_stride_0, push.input_stride_1,
		push.input_stride_2, push.input_stride_3
	};
	uint remainder = index;
	uint input_offset = 0;
	for (int axis = int(push.rank) - 1; axis >= 0; --axis) {
		uint output_extent = input_dims[axis] * ((axis == int(push.dim)) ? push.repeats : 1);
		uint coordinate = remainder % output_extent;
		remainder /= output_extent;
		uint input_coordinate =
			(axis == int(push.dim)) ? coordinate / push.repeats : coordinate;
		input_offset += input_coordinate * input_strides[axis];
	}
	float value = load_f32(storage_buffers[push.input_index], input_offset);
	store_f32(storage_buffers[push.output_index], index, value);
}
