// Mechanical adaptation of OA loss/maskedMse.slang.

import storage;
import attributes;

struct PushConstants {
	uint prediction_index;
	uint target_index;
	uint mask_index;
	uint loss_index;
	uint denominator_index;
	uint element_count;
	uint rank;
	uint prediction_stride_0; uint prediction_stride_1; uint prediction_stride_2; uint prediction_stride_3;
	uint prediction_stride_4; uint prediction_stride_5; uint prediction_stride_6; uint prediction_stride_7;
	uint mask_dim_0; uint mask_dim_1; uint mask_dim_2; uint mask_dim_3;
	uint mask_dim_4; uint mask_dim_5; uint mask_dim_6; uint mask_dim_7;
	uint mask_stride_0; uint mask_stride_1; uint mask_stride_2; uint mask_stride_3;
	uint mask_stride_4; uint mask_stride_5; uint mask_stride_6; uint mask_stride_7;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float group_error[256];
groupshared float group_weight[256];

[ForceInline]
uint predictionStride(uint dim) {
	if (dim == 0u) return push.prediction_stride_0;
	if (dim == 1u) return push.prediction_stride_1;
	if (dim == 2u) return push.prediction_stride_2;
	if (dim == 3u) return push.prediction_stride_3;
	if (dim == 4u) return push.prediction_stride_4;
	if (dim == 5u) return push.prediction_stride_5;
	if (dim == 6u) return push.prediction_stride_6;
	return push.prediction_stride_7;
}

[ForceInline]
uint maskDim(uint dim) {
	if (dim == 0u) return push.mask_dim_0;
	if (dim == 1u) return push.mask_dim_1;
	if (dim == 2u) return push.mask_dim_2;
	if (dim == 3u) return push.mask_dim_3;
	if (dim == 4u) return push.mask_dim_4;
	if (dim == 5u) return push.mask_dim_5;
	if (dim == 6u) return push.mask_dim_6;
	return push.mask_dim_7;
}

[ForceInline]
uint maskStride(uint dim) {
	if (dim == 0u) return push.mask_stride_0;
	if (dim == 1u) return push.mask_stride_1;
	if (dim == 2u) return push.mask_stride_2;
	if (dim == 3u) return push.mask_stride_3;
	if (dim == 4u) return push.mask_stride_4;
	if (dim == 5u) return push.mask_stride_5;
	if (dim == 6u) return push.mask_stride_6;
	return push.mask_stride_7;
}

[ForceInline]
uint maskIndex(uint flat_index) {
	uint remaining = flat_index;
	uint mask_index = 0u;
	[unroll] for (uint dim = 0u; dim < 8u; ++dim) {
		if (dim >= push.rank) {
			break;
		}
		uint stride = predictionStride(dim);
		uint coordinate = remaining / stride;
		remaining -= coordinate * stride;
		if (maskDim(dim) != 1u) {
			mask_index += coordinate * maskStride(dim);
		}
	}
	return mask_index;
}

[kernel_name("masked_mse")]
[domain("ml")]
[variant("broadcast_mask")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float error_sum = 0.0f;
	float weight_sum = 0.0f;
	for (uint index = lane; index < push.element_count; index += 256u) {
		float weight = load_f32(storage_buffers[push.mask_index], maskIndex(index));
		float difference = load_f32(storage_buffers[push.prediction_index], index)
			- load_f32(storage_buffers[push.target_index], index);
		error_sum += difference * difference * weight;
		weight_sum += weight;
	}
	group_error[lane] = error_sum;
	group_weight[lane] = weight_sum;
	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			group_error[lane] += group_error[lane + stride];
			group_weight[lane] += group_weight[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0u) {
		float denominator = max(group_weight[0], 1.0f);
		store_f32(storage_buffers[push.loss_index], 0u, group_error[0] / denominator);
		store_f32(storage_buffers[push.denominator_index], 0u, denominator);
	}
}
