// Composition adaptation of donor splitHeads, scalar BMM, sdpaSoftmax and
// mergeHeads. Preserve all source outputs and ascending dot/value sums.
// One workgroup owns a batch/head; up to eight query rows share cached Q/K/V.
import storage;
import attributes;
struct PushConstants {
	uint query_index; uint key_index; uint value_index;
	uint split_query_index; uint split_key_index; uint split_value_index;
	uint scores_index; uint probabilities_index; uint context_index; uint output_index;
	uint batch; uint sequence_length; uint num_heads; uint head_dim; float scale;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
static const uint MAX_ELEMENTS = 2048u;
groupshared float queries[MAX_ELEMENTS];
groupshared float keys[MAX_ELEMENTS];
groupshared float values[MAX_ELEMENTS];
groupshared float weights[1024];
[kernel_name("small_causal_sdpa")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint bh = group_id.x;
	uint thread = group_thread_id.x;
	if (bh >= push.batch * push.num_heads) return;
	uint batch = bh / push.num_heads;
	uint head = bh % push.num_heads;
	uint count = push.sequence_length * push.head_dim;
	for (uint i = thread; i < count; i += 256u) {
		uint sequence = i / push.head_dim;
		uint d = i % push.head_dim;
		uint packed = ((batch * push.sequence_length + sequence) * push.num_heads + head) * push.head_dim + d;
		float q = load_f32(storage_buffers[push.query_index], packed);
		float k = load_f32(storage_buffers[push.key_index], packed);
		float v = load_f32(storage_buffers[push.value_index], packed);
		queries[i] = q; keys[i] = k; values[i] = v;
		store_f32(storage_buffers[push.split_query_index], bh * count + i, q);
		store_f32(storage_buffers[push.split_key_index], bh * count + i, k);
		store_f32(storage_buffers[push.split_value_index], bh * count + i, v);
	}
	GroupMemoryBarrierWithGroupSync();
	// One lane per key column admits every sequence the matcher passes; for
	// sequence <= 16 the whole query set fits one tile, so a workgroup pays
	// the staging barrier plus the two tile barriers exactly once.
	uint row_lanes = min(push.sequence_length, 128u);
	uint queries_per_tile = 256u / row_lanes;
	uint slot = thread / row_lanes;
	uint lane = thread % row_lanes;
	uint offset = slot * row_lanes;
	for (uint first = 0u; first < push.sequence_length; first += queries_per_tile) {
		uint query = first + slot;
		bool active = query < push.sequence_length && lane < push.sequence_length;
		float score = -3.402823466e+38f;
		if (active) {
			float dot = 0.0f;
			for (uint d = 0u; d < push.head_dim; ++d) dot += queries[query * push.head_dim + d] * keys[lane * push.head_dim + d];
			store_f32(storage_buffers[push.scores_index], (bh * push.sequence_length + query) * push.sequence_length + lane, dot);
			score = dot * push.scale;
			if (lane > query) score = -3.402823466e+38f;
		}
		// At most 128 masked scores per row; a serial row softmax in one
		// lane costs less than two shared-memory tree reductions on this
		// device class, where every stride level is a workgroup barrier.
		weights[offset + lane] = score;
		GroupMemoryBarrierWithGroupSync();
		if (lane == 0u && query < push.sequence_length) {
			float maximum = weights[offset];
			for (uint key = 1u; key < push.sequence_length; ++key) maximum = max(maximum, weights[offset + key]);
			float sum = 0.0f;
			for (uint key = 0u; key < push.sequence_length; ++key) {
				float exponent = exp(weights[offset + key] - maximum);
				weights[offset + key] = exponent;
				sum += exponent;
			}
			float inverse = 1.0f / sum;
			uint row_index = (bh * push.sequence_length + query) * push.sequence_length;
			for (uint key = 0u; key < push.sequence_length; ++key) {
				float probability = weights[offset + key] * inverse;
				weights[offset + key] = probability;
				store_f32(storage_buffers[push.probabilities_index], row_index + key, probability);
			}
		}
		GroupMemoryBarrierWithGroupSync();
		if (query < push.sequence_length) {
			for (uint d = lane; d < push.head_dim; d += row_lanes) {
				float result = 0.0f;
				for (uint key = 0u; key < push.sequence_length; ++key) result += weights[offset + key] * values[key * push.head_dim + d];
				store_f32(storage_buffers[push.context_index], (bh * push.sequence_length + query) * push.head_dim + d, result);
				uint packed = ((batch * push.sequence_length + query) * push.num_heads + head) * push.head_dim + d;
				store_f32(storage_buffers[push.output_index], packed, result);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
