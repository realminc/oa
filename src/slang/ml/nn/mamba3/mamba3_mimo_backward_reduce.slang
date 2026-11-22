// Deterministic reductions for the MIMO adjoint: head-to-group Q/K gradients,
// shared angle gradients, and batch partials for all learned parameters.

import storage;
import attributes;

struct PushConstants {
	uint dc_head_idx, db_head_idx, dangle_head_idx;
	uint dc_bias_batch_idx, db_bias_batch_idx, dd_batch_idx;
	uint dmx_batch_idx, dmz_batch_idx, dmo_batch_idx, dnorm_batch_idx;
	uint dc_idx, db_idx, dangle_idx, dc_bias_idx, db_bias_idx, dd_idx;
	uint dmx_idx, dmz_idx, dmo_idx, dnorm_idx;
	uint b, l, h, g, p, n, a, r, count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_mimo_backward_reduce")]
[domain("ml")]
[variant("reduce")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= push.count) return;
	uint groupCount = push.b * push.l * push.r * push.g * push.n;
	uint angleCount = push.b * push.l * push.a;
	uint biasCount = push.h * push.r * push.n;
	uint mimoCount = push.h * push.r * push.p;
	uint normCount = push.h * push.p;
	uint cursor = 0u;
	if (index < groupCount) {
		uint n = index % push.n;
		uint q = index / push.n;
		uint rg = q % (push.r * push.g);
		uint token = q / (push.r * push.g);
		uint r = rg / push.g;
		uint g = rg % push.g;
		uint headsPerGroup = push.h / push.g;
		float dc = 0.0f, db = 0.0f;
		for (uint local = 0u; local < headsPerGroup; ++local) {
			uint h = g * headsPerGroup + local;
			uint headIndex = ((token * push.h + h) * push.r + r) * push.n + n;
			dc += load_f32(storage_buffers[push.dc_head_idx], headIndex);
			db += load_f32(storage_buffers[push.db_head_idx], headIndex);
		}
		store_f32(storage_buffers[push.dc_idx], index, dc);
		store_f32(storage_buffers[push.db_idx], index, db);
		return;
	}
	cursor += groupCount;
	if (index < cursor + angleCount) {
		uint localIndex = index - cursor;
		uint a = localIndex % push.a;
		uint token = localIndex / push.a;
		uint t = token % push.l;
		uint b = token / push.l;
		float value = 0.0f;
		for (uint h = 0u; h < push.h; ++h)
			value += load_f32(storage_buffers[push.dangle_head_idx], ((b * push.h + h) * push.l + t) * push.a + a);
		store_f32(storage_buffers[push.dangle_idx], localIndex, value);
		return;
	}
	cursor += angleCount;
	if (index < cursor + biasCount) {
		uint localIndex = index - cursor;
		float dc = 0.0f, db = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			uint partial = b * biasCount + localIndex;
			dc += load_f32(storage_buffers[push.dc_bias_batch_idx], partial);
			db += load_f32(storage_buffers[push.db_bias_batch_idx], partial);
		}
		store_f32(storage_buffers[push.dc_bias_idx], localIndex, dc);
		store_f32(storage_buffers[push.db_bias_idx], localIndex, db);
		return;
	}
	cursor += biasCount;
	if (index < cursor + push.h) {
		uint h = index - cursor;
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b)
			value += load_f32(storage_buffers[push.dd_batch_idx], b * push.h + h);
		store_f32(storage_buffers[push.dd_idx], h, value);
		return;
	}
	cursor += push.h;
	if (index < cursor + mimoCount) {
		uint localIndex = index - cursor;
		float dx = 0.0f, dz = 0.0f, dout = 0.0f;
		for (uint b = 0u; b < push.b; ++b) {
			uint partial = b * mimoCount + localIndex;
			dx += load_f32(storage_buffers[push.dmx_batch_idx], partial);
			dz += load_f32(storage_buffers[push.dmz_batch_idx], partial);
			dout += load_f32(storage_buffers[push.dmo_batch_idx], partial);
		}
		store_f32(storage_buffers[push.dmx_idx], localIndex, dx);
		store_f32(storage_buffers[push.dmz_idx], localIndex, dz);
		store_f32(storage_buffers[push.dmo_idx], localIndex, dout);
		return;
	}
	cursor += mimoCount;
	uint normIndex = index - cursor;
	if (normIndex < normCount) {
		float value = 0.0f;
		for (uint b = 0u; b < push.b; ++b)
			value += load_f32(storage_buffers[push.dnorm_batch_idx], b * normCount + normIndex);
		store_f32(storage_buffers[push.dnorm_idx], normIndex, value);
	}
}
