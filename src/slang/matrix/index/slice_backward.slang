// Donor: source/cpp/lib/oa/core/shader/compute/flat/matrixCopyRegion.slang
// Pad one rank-1..4 FP32 slice adjoint into its original shape.

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint rank;
	uint dim;
	uint source_start;
	uint destination_start;
	uint source_dim_0;
	uint source_dim_1;
	uint source_dim_2;
	uint source_dim_3;
	uint destination_dim_0;
	uint destination_dim_1;
	uint destination_dim_2;
	uint destination_dim_3;
	uint copy_dim_0;
	uint copy_dim_1;
	uint copy_dim_2;
	uint copy_dim_3;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("slice_backward")]
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

	uint source_dims[4] = {
		push.source_dim_0, push.source_dim_1, push.source_dim_2, push.source_dim_3
	};
	uint destination_dims[4] = {
		push.destination_dim_0, push.destination_dim_1,
		push.destination_dim_2, push.destination_dim_3
	};
	uint copy_dims[4] = {
		push.copy_dim_0, push.copy_dim_1, push.copy_dim_2, push.copy_dim_3
	};
	uint coordinates[4] = { 0, 0, 0, 0 };
	uint remainder = index;
	for (int axis = int(push.rank) - 1; axis >= 0; --axis) {
		coordinates[axis] = remainder % copy_dims[axis];
		remainder /= copy_dims[axis];
	}

	uint source_offset = 0;
	uint destination_offset = 0;
	for (uint axis = 0; axis < push.rank; ++axis) {
		uint source_coordinate =
			coordinates[axis] + ((axis == push.dim) ? push.source_start : 0);
		uint destination_coordinate =
			coordinates[axis] + ((axis == push.dim) ? push.destination_start : 0);
		source_offset = source_offset * source_dims[axis] + source_coordinate;
		destination_offset =
			destination_offset * destination_dims[axis] + destination_coordinate;
	}
	float value = load_f32(storage_buffers[push.input_index], source_offset);
	store_f32(storage_buffers[push.output_index], destination_offset, value);
}
