// Deterministic completion for the bounded generic SISO adjoint.

import storage;
import activations;
import attributes;

struct PushConstants {
	uint dc_head_idx, db_head_idx, dangle_head_idx, ddtok_idx;
	uint trap_idx, dtrap_partial_idx;
	uint dc_idx, db_idx, dangle_idx, cbias_idx, bbias_idx, dd_idx, dtrap_idx;
	uint b, l, h, g, n, a, count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_generic_reduce")]
[domain("ml")]
[variant("deterministic_reduce")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
	uint index = dispatch_id.x;
	if (index >= push.count) return;
	uint group_count = push.b * push.l * push.g * push.n;
	uint angle_count = push.b * push.l * push.a;
	uint bias_count = push.h * push.n;
	uint trap_count = push.b * push.l * push.h;
	uint cursor = 0u;
	if (index < group_count) {
		uint n = index % push.n;
		uint token_group = index / push.n;
		uint g = token_group % push.g;
		uint token = token_group / push.g;
		uint heads_per_group = push.h / push.g;
		float dc = 0.0f, db = 0.0f;
		for (uint local = 0u; local < heads_per_group; ++local) {
			uint h = g * heads_per_group + local;
			uint source = (token * push.h + h) * push.n + n;
			dc += load_f32(storage_buffers[push.dc_head_idx], source);
			db += load_f32(storage_buffers[push.db_head_idx], source);
		}
		store_f32(storage_buffers[push.dc_idx], index, dc);
		store_f32(storage_buffers[push.db_idx], index, db);
		return;
	}
	cursor += group_count;
	if (index < cursor + angle_count) {
		uint local_index = index - cursor;
		uint a = local_index % push.a;
		uint token = local_index / push.a;
		uint t = token % push.l;
		uint b = token / push.l;
		float value = 0.0f;
		for (uint h = 0u; h < push.h; ++h)
			value += load_f32(storage_buffers[push.dangle_head_idx],
				((b * push.h + h) * push.l + t) * push.a + a);
		store_f32(storage_buffers[push.dangle_idx], local_index, value);
		return;
	}
	cursor += angle_count;
	if (index < cursor + bias_count) {
		uint local_index = index - cursor;
		uint h = local_index / push.n;
		uint n = local_index % push.n;
		float dc = 0.0f, db = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			for (uint t = 0u; t < push.l; ++t) {
				uint source = ((b * push.l + t) * push.h + h) * push.n + n;
				dc += load_f32(storage_buffers[push.dc_head_idx], source);
				db += load_f32(storage_buffers[push.db_head_idx], source);
			}
		}
		store_f32(storage_buffers[push.cbias_idx], local_index, dc);
		store_f32(storage_buffers[push.bbias_idx], local_index, db);
		return;
	}
	cursor += bias_count;
	if (index < cursor + push.h) {
		uint h = index - cursor;
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b)
			for (uint t = 0u; t < push.l; ++t)
				value += load_f32(storage_buffers[push.ddtok_idx],
					(b * push.l + t) * push.h + h);
		store_f32(storage_buffers[push.dd_idx], h, value);
		return;
	}
	cursor += push.h;
	uint scalar = index - cursor;
	if (scalar < trap_count) {
		float probability = sigmoid(load_f32(storage_buffers[push.trap_idx], scalar));
		float partial = load_f32(storage_buffers[push.dtrap_partial_idx], scalar);
		store_f32(storage_buffers[push.dtrap_idx], scalar,
			partial * probability * (1.0f - probability));
	}
}
