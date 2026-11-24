// Pad a rank-1..5 FP32 Matrix per dimension. Mode selects the border
// contract: constant fill, replicate (edge clamp), reflect (mirror without
// repeating the edge), or circular (wrap). pad_before is the leading pad per
// dimension; the trailing pad is implicit in the shapes.

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint rank;
	uint mode;
	uint fill_bits;
	uint in_dim_0;
	uint in_dim_1;
	uint in_dim_2;
	uint in_dim_3;
	uint in_dim_4;
	uint out_dim_0;
	uint out_dim_1;
	uint out_dim_2;
	uint out_dim_3;
	uint out_dim_4;
	uint pad_0;
	uint pad_1;
	uint pad_2;
	uint pad_3;
	uint pad_4;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint source_coordinate(int coordinate, uint extent, uint pad_before) {
	if (push.mode == 1) {
		int shifted = coordinate - int(pad_before);
		return uint(clamp(shifted, 0, int(extent) - 1));
	}
	if (push.mode == 2) {
		int shifted = coordinate - int(pad_before);
		int last = int(extent) - 1;
		for (int iteration = 0; iteration < 8; ++iteration) {
			if (shifted < 0) {
				shifted = -shifted;
			} else if (shifted > last) {
				shifted = 2 * last - shifted;
			} else {
				break;
			}
		}
		return uint(shifted);
	}
	if (push.mode == 3) {
		int shifted = (coordinate - int(pad_before)) % int(extent);
		return uint(shifted < 0 ? shifted + int(extent) : shifted);
	}
	return uint(coordinate) - pad_before;
}

[kernel_name("pad")]
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
	uint in_dims[5] = {
		push.in_dim_0, push.in_dim_1, push.in_dim_2, push.in_dim_3, push.in_dim_4
	};
	uint out_dims[5] = {
		push.out_dim_0, push.out_dim_1, push.out_dim_2, push.out_dim_3,
		push.out_dim_4
	};
	uint pads[5] = { push.pad_0, push.pad_1, push.pad_2, push.pad_3, push.pad_4 };
	uint coordinates[5] = { 0, 0, 0, 0, 0 };
	uint remainder = index;
	for (int axis = int(push.rank) - 1; axis >= 0; --axis) {
		coordinates[axis] = remainder % out_dims[axis];
		remainder /= out_dims[axis];
	}
	uint source_index = 0;
	bool inside = true;
	for (uint axis = 0; axis < push.rank; ++axis) {
		int coordinate = int(coordinates[axis]);
		if (
			push.mode == 0
			&& (coordinate < int(pads[axis])
				|| coordinate >= int(pads[axis] + in_dims[axis]))
		) {
			inside = false;
			break;
		}
		source_index = source_index * in_dims[axis]
			+ source_coordinate(coordinate, in_dims[axis], pads[axis]);
	}
	float value = inside
		? load_f32(storage_buffers[push.input_index], source_index)
		: asfloat(push.fill_bits);
	store_f32(storage_buffers[push.output_index], index, value);
}
