// Donors: source/cpp/lib/oa/ml/shader/compute/nn/softmax/softmax.slang,
// source/cpp/lib/oa/core/shader/compute/flat/{topK,topKMask}.slang.
// Composition adaptation: unchanged softmax and argmax reduction order;
// one workgroup per row materializes every source output. Selection logits
// may carry balancing bias; probabilities always use the unbiased logits.
import storage;
import attributes;

struct PushConstants {
	uint logits_index;
	uint selection_index;
	uint probabilities_index;
	uint values_index;
	uint indices_index;
	uint mask_index;
	uint rows;
	uint experts;
	uint k;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
static const uint THREADS = 256;
static const uint MAX_K = 1024;
groupshared float shared_values[THREADS];
groupshared uint shared_indices[THREADS];
groupshared uint taken[MAX_K];

[kernel_name("moe_router_select")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint group = group_id.x;
	uint lane = group_thread_id.x;
	if (group >= push.rows) {
		return;
	}

	uint base = group * push.experts;

	float local_maximum = -3.402823466e+38f;
	for (uint index = lane; index < push.experts; index += 256) {
		local_maximum = max(
			local_maximum,
			load_f32(storage_buffers[push.logits_index], base + index)
		);
	}
	shared_values[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] = max(shared_values[lane], shared_values[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = shared_values[0];
	GroupMemoryBarrierWithGroupSync();

	float local_sum = 0.0f;
	for (uint index = lane; index < push.experts; index += 256) {
		local_sum += exp(
			load_f32(storage_buffers[push.logits_index], base + index)
				- row_maximum
		);
	}
	shared_values[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] += shared_values[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_sum = 1.0f / shared_values[0];
	GroupMemoryBarrierWithGroupSync();

	for (uint index = lane; index < push.experts; index += 256) {
		uint element = base + index;
		store_f32(
			storage_buffers[push.probabilities_index],
			element,
			exp(load_f32(storage_buffers[push.logits_index], element) - row_maximum)
				* inverse_sum
		);
	}

	uint row = group;
	GroupMemoryBarrierWithGroupSync();

	for (uint rank = 0; rank < push.k; ++rank) {
		float best_value = -3.402823466e+38f;
		uint best_index = 0xffffffffu;
		for (uint index = lane; index < push.experts; index += THREADS) {
			bool is_taken = false;
			for (uint previous = 0; previous < rank; ++previous) {
				if (taken[previous] == index) { is_taken = true; break; }
			}
			if (is_taken) continue;
			float value = load_f32(storage_buffers[push.selection_index], base + index);
			if (value > best_value || (value == best_value && index < best_index)) {
				best_value = value;
				best_index = index;
			}
		}

		shared_values[lane] = best_value;
		shared_indices[lane] = best_index;
		GroupMemoryBarrierWithGroupSync();
		for (uint stride = 128; stride > 0; stride >>= 1) {
			if (lane < stride &&
				(shared_values[lane + stride] > shared_values[lane] ||
				(shared_values[lane + stride] == shared_values[lane] &&
				 shared_indices[lane + stride] < shared_indices[lane]))) {
				shared_values[lane] = shared_values[lane + stride];
				shared_indices[lane] = shared_indices[lane + stride];
			}
			GroupMemoryBarrierWithGroupSync();
		}
		if (lane == 0) {
			taken[rank] = shared_indices[0];
			store_f32(storage_buffers[push.values_index], row * push.k + rank, shared_values[0]);
			storage_buffers[push.indices_index].Store((row * push.k + rank) * 4, shared_indices[0]);
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint index = lane; index < push.experts; index += THREADS) {
		float selected = 0.0f;
		for (uint rank = 0; rank < push.k; ++rank) {
			if (taken[rank] == index) { selected = 1.0f; break; }
		}
		store_f32(storage_buffers[push.mask_index], base + index, selected);
	}
}
