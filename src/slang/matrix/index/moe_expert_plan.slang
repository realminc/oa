// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeExpertPlan.slang
// Stable dropless expert-major route packing in token-major input order.

import attributes;

struct PushConstants {
	uint indices_index;
	uint counts_index;
	uint offsets_index;
	uint packed_token_index;
	uint packed_expert_index;
	uint packed_slot_index;
	uint inverse_index;
	uint tokens;
	uint routes_per_token;
	uint experts;
	uint routes;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint THREADS = 256;
groupshared uint counts[THREADS];
groupshared uint cursors[THREADS];
groupshared uint scan_a[THREADS];
groupshared uint scan_b[THREADS];
groupshared uint expert_base;
groupshared uint4 scan4_a[THREADS];
groupshared uint4 scan4_b[THREADS];
groupshared uint expert_bases4[4];

[kernel_name("moe_expert_plan")]
[domain("matrix")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	if (lane < push.experts) counts[lane] = 0;
	GroupMemoryBarrierWithGroupSync();

	for (uint route = lane; route < push.routes; route += THREADS) {
		uint expert = storage_buffers[push.indices_index].Load(route * 4);
		if (expert < push.experts) {
			uint ignored;
			InterlockedAdd(counts[expert], 1, ignored);
		}
	}
	GroupMemoryBarrierWithGroupSync();

	if (lane == 0) {
		uint offset = 0;
		for (uint expert = 0; expert < push.experts; ++expert) {
			uint count = counts[expert];
			storage_buffers[push.counts_index].Store(expert * 4, count);
			storage_buffers[push.offsets_index].Store(expert * 4, offset);
			cursors[expert] = offset;
			offset += count;
		}
		storage_buffers[push.offsets_index].Store(push.experts * 4, offset);
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint base = 0; base < push.routes; base += THREADS) {
		uint route = base + lane;
		uint route_expert = route < push.routes
			? storage_buffers[push.indices_index].Load(route * 4) : push.experts;
		if (push.experts <= 4) {
			scan4_a[lane] = uint4(
				route_expert == 0 ? 1u : 0u,
				route_expert == 1 ? 1u : 0u,
				route_expert == 2 ? 1u : 0u,
				route_expert == 3 ? 1u : 0u);
			GroupMemoryBarrierWithGroupSync();
			bool in_a4 = true;
			for (uint offset = 1; offset < THREADS; offset <<= 1) {
				uint4 value = in_a4 ? scan4_a[lane] : scan4_b[lane];
				if (lane >= offset) value += in_a4 ? scan4_a[lane - offset] : scan4_b[lane - offset];
				if (in_a4) scan4_b[lane] = value; else scan4_a[lane] = value;
				GroupMemoryBarrierWithGroupSync();
				in_a4 = !in_a4;
			}
			uint4 inclusive4 = in_a4 ? scan4_a[lane] : scan4_b[lane];
			uint4 chunk_counts4 = in_a4 ? scan4_a[THREADS - 1] : scan4_b[THREADS - 1];
			if (lane < push.experts) {
				uint chunk_count = lane == 0 ? chunk_counts4.x
					: lane == 1 ? chunk_counts4.y : lane == 2 ? chunk_counts4.z : chunk_counts4.w;
				expert_bases4[lane] = cursors[lane];
				cursors[lane] += chunk_count;
			}
			GroupMemoryBarrierWithGroupSync();
			if (route < push.routes && route_expert < push.experts) {
				uint inclusive_rank = route_expert == 0 ? inclusive4.x
					: route_expert == 1 ? inclusive4.y : route_expert == 2 ? inclusive4.z : inclusive4.w;
				uint packed = expert_bases4[route_expert] + inclusive_rank - 1;
				storage_buffers[push.packed_token_index].Store(packed * 4, route / push.routes_per_token);
				storage_buffers[push.packed_expert_index].Store(packed * 4, route_expert);
				storage_buffers[push.packed_slot_index].Store(packed * 4, route);
				storage_buffers[push.inverse_index].Store(route * 4, packed);
			}
			continue;
		}
		for (uint expert = 0; expert < push.experts; ++expert) {
			uint selected = route_expert == expert ? 1 : 0;
			scan_a[lane] = selected;
			GroupMemoryBarrierWithGroupSync();
			bool in_a = true;
			for (uint offset = 1; offset < THREADS; offset <<= 1) {
				uint value = in_a ? scan_a[lane] : scan_b[lane];
				if (lane >= offset) value += in_a ? scan_a[lane - offset] : scan_b[lane - offset];
				if (in_a) scan_b[lane] = value; else scan_a[lane] = value;
				GroupMemoryBarrierWithGroupSync();
				in_a = !in_a;
			}
			uint inclusive_rank = in_a ? scan_a[lane] : scan_b[lane];
			uint chunk_count = in_a ? scan_a[THREADS - 1] : scan_b[THREADS - 1];
			if (lane == 0) { expert_base = cursors[expert]; cursors[expert] += chunk_count; }
			GroupMemoryBarrierWithGroupSync();
			if (selected != 0) {
				uint packed = expert_base + inclusive_rank - 1;
				storage_buffers[push.packed_token_index].Store(packed * 4, route / push.routes_per_token);
				storage_buffers[push.packed_expert_index].Store(packed * 4, expert);
				storage_buffers[push.packed_slot_index].Store(packed * 4, route);
				storage_buffers[push.inverse_index].Store(route * 4, packed);
			}
		}
	}
}
