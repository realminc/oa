// Donor: source/cpp/lib/oa/core/shader/compute/flat/topK.slang
// Mechanical adaptation: Rust naming, OA attributes, and storage helpers.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint values_index;
	uint indices_index;
	uint rows;
	uint row_width;
	uint k;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint THREADS = 256;
static const uint MAX_K = 1024;

groupshared float shared_values[THREADS];
groupshared uint shared_indices[THREADS];
groupshared uint taken[MAX_K];

[kernel_name("top_k")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	if (row >= push.rows) return;
	uint lane = group_thread_id.x;
	uint base = row * push.row_width;

	for (uint rank = 0; rank < push.k; ++rank) {
		float best_value = -3.402823466e+38f;
		uint best_index = 0xffffffffu;
		for (uint index = lane; index < push.row_width; index += THREADS) {
			bool is_taken = false;
			for (uint previous = 0; previous < rank; ++previous) {
				if (taken[previous] == index) { is_taken = true; break; }
			}
			if (is_taken) continue;
			float value = load_f32(storage_buffers[push.input_index], base + index);
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
}
