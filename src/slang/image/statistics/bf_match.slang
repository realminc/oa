import attributes;
import storage;

struct PushConstants {
	uint query_index, train_index, output_index;
	uint query_count, train_count, desc_len;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Brute-force Hamming matcher: one thread per query descriptor scans the
// train set and writes (best index, best distance, second index, second
// distance) — the knnMatch(k=2) contract.
[kernel_name("bf_match")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint query = thread_id.x;
	if (query >= push.query_count) return;
	RWByteAddressBuffer query_buffer = storage_buffers[push.query_index];
	RWByteAddressBuffer train_buffer = storage_buffers[push.train_index];
	uint best_index = 0xFFFFFFFFu;
	uint best_dist = 0xFFFFFFFFu;
	uint second_index = 0xFFFFFFFFu;
	uint second_dist = 0xFFFFFFFFu;
	for (uint t = 0u; t < push.train_count; t++) {
		uint dist = 0u;
		for (uint b = 0u; b < push.desc_len; b++) {
			uint qb = load_u8(query_buffer, query * push.desc_len + b);
			uint tb = load_u8(train_buffer, t * push.desc_len + b);
			dist += countbits(qb ^ tb);
		}
		if (dist < best_dist) {
			second_index = best_index;
			second_dist = best_dist;
			best_index = t;
			best_dist = dist;
		} else if (dist < second_dist) {
			second_index = t;
			second_dist = dist;
		}
	}
	RWByteAddressBuffer output = storage_buffers[push.output_index];
	uint base = query * 4;
	store_f32(output, base, best_index == 0xFFFFFFFFu ? -1.0 : float(best_index));
	store_f32(output, base + 1, best_dist == 0xFFFFFFFFu ? -1.0 : float(best_dist));
	store_f32(output, base + 2, second_index == 0xFFFFFFFFu ? -1.0 : float(second_index));
	store_f32(output, base + 3, second_dist == 0xFFFFFFFFu ? -1.0 : float(second_dist));
}
