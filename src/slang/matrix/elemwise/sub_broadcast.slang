// Mechanical adaptation of OA flat/subBcast.slang.
// General rank-eight multidirectional FP32 broadcast subtraction.

import attributes;
import storage;

struct PushConstants {
	uint left_index;
	uint right_index;
	uint output_index;
	uint element_count;
	uint rank;
	uint output_dim_0;
	uint output_dim_1;
	uint output_dim_2;
	uint output_dim_3;
	uint output_dim_4;
	uint output_dim_5;
	uint output_dim_6;
	uint output_dim_7;
	uint left_stride_0;
	uint left_stride_1;
	uint left_stride_2;
	uint left_stride_3;
	uint left_stride_4;
	uint left_stride_5;
	uint left_stride_6;
	uint left_stride_7;
	uint right_stride_0;
	uint right_stride_1;
	uint right_stride_2;
	uint right_stride_3;
	uint right_stride_4;
	uint right_stride_5;
	uint right_stride_6;
	uint right_stride_7;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sub_broadcast")]
[domain("matrix")]
[variant("broadcast")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	uint output_dims[8] = {
		push.output_dim_0, push.output_dim_1, push.output_dim_2, push.output_dim_3,
		push.output_dim_4, push.output_dim_5, push.output_dim_6, push.output_dim_7
	};
	uint left_strides[8] = {
		push.left_stride_0, push.left_stride_1, push.left_stride_2, push.left_stride_3,
		push.left_stride_4, push.left_stride_5, push.left_stride_6, push.left_stride_7
	};
	uint right_strides[8] = {
		push.right_stride_0, push.right_stride_1, push.right_stride_2, push.right_stride_3,
		push.right_stride_4, push.right_stride_5, push.right_stride_6, push.right_stride_7
	};
	uint remainder = index;
	uint left_offset = 0;
	uint right_offset = 0;
	for (int axis = int(push.rank) - 1; axis >= 0; --axis) {
		uint coordinate = remainder % output_dims[axis];
		remainder /= output_dims[axis];
		left_offset += coordinate * left_strides[axis];
		right_offset += coordinate * right_strides[axis];
	}
	float left = load_f32(storage_buffers[push.left_index], left_offset);
	float right = load_f32(storage_buffers[push.right_index], right_offset);
	store_f32(storage_buffers[push.output_index], index, left - right);
}
