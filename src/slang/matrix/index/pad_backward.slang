// Input adjoint of `pad`: every output element has exactly one source (or the
// fill constant), so each input element accumulates the output gradients of
// its preimage set. Constant mode's preimage is a single coordinate; the
// border modes add at most two extra wrapped/mirrored ranges per dimension.

import attributes;
import storage;

struct PushConstants {
	uint gradient_index;
	uint input_gradient_index;
	uint count;
	uint rank;
	uint mode;
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

// Emit up to three half-open output-coordinate ranges [start, start + count)
// that map onto input coordinate `j` for one dimension.
uint preimage_ranges(
	uint j,
	uint n,
	uint out_extent,
	uint pad_before,
	uint mode,
	out uint start0,
	out uint start1,
	out uint start2,
	out uint count0,
	out uint count1,
	out uint count2
) {
	start0 = j + pad_before;
	count0 = 1;
	start1 = 0;
	start2 = 0;
	count1 = 0;
	count2 = 0;
	uint pad_after = out_extent - n - pad_before;
	if (mode == 1) {
		if (j == 0) {
			start0 = 0;
			count0 = pad_before + 1;
		} else if (j == n - 1) {
			count0 = pad_after + 1;
		}
	} else if (mode == 2) {
		if (j > 0 && j <= pad_before) {
			start1 = pad_before - j;
			count1 = 1;
		}
		if (j < n - 1 && j + pad_after >= n - 1) {
			start2 = 2 * (n - 1) - j + pad_before;
			count2 = 1;
		}
	} else if (mode == 3) {
		if (j + pad_before >= n) {
			start1 = j + pad_before - n;
			count1 = 1;
		}
		if (j < pad_after) {
			start2 = j + pad_before + n;
			count2 = 1;
		}
	}
	return count0 + count1 + count2;
}

[kernel_name("pad_backward")]
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
		coordinates[axis] = remainder % in_dims[axis];
		remainder /= in_dims[axis];
	}
	uint start0[5];
	uint start1[5];
	uint start2[5];
	uint count0[5];
	uint count1[5];
	uint count2[5];
	uint total = 1;
	for (uint axis = 0; axis < 5; ++axis) {
		if (axis < push.rank) {
			uint candidates = preimage_ranges(
				coordinates[axis], in_dims[axis], out_dims[axis], pads[axis],
				push.mode, start0[axis], start1[axis], start2[axis],
				count0[axis], count1[axis], count2[axis]
			);
			total *= candidates;
		} else {
			start0[axis] = 0;
			start1[axis] = 0;
			start2[axis] = 0;
			count0[axis] = 1;
			count1[axis] = 0;
			count2[axis] = 0;
		}
	}
	float gradient = 0.0;
	for (uint combination = 0; combination < total; ++combination) {
		uint stride = 1;
		uint source = 0;
		for (uint axis = 0; axis < push.rank; ++axis) {
			uint options = count0[axis] + count1[axis] + count2[axis];
			uint choice = (combination / stride) % options;
			uint coordinate;
			if (choice < count0[axis]) {
				coordinate = start0[axis] + choice;
			} else if (choice < count0[axis] + count1[axis]) {
				coordinate = start1[axis] + choice - count0[axis];
			} else {
				coordinate = start2[axis] + choice - count0[axis] - count1[axis];
			}
			source = source * out_dims[axis] + coordinate;
			stride *= options;
		}
		gradient += load_f32(storage_buffers[push.gradient_index], source);
	}
	store_f32(storage_buffers[push.input_gradient_index], index, gradient);
}
