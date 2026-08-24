// Deterministic reductions for the MIMO adjoint: head-to-group Q/K gradients,
// shared angle gradients, and batch partials for all learned parameters.

import storage;

struct PushConstants {
	uint dc_head_idx, db_head_idx, dangle_head_idx;
	uint dc_bias_batch_idx, db_bias_batch_idx, dd_batch_idx;
	uint dmx_batch_idx, dmz_batch_idx, dmo_batch_idx, dnorm_batch_idx;
	uint dc_idx, db_idx, dangle_idx, dc_bias_idx, db_bias_idx, dd_idx;
	uint dmx_idx, dmz_idx, dmo_idx, dnorm_idx;
	uint B, L, H, G, P, N, A, R, count;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= pc.count) return;
	uint groupCount = pc.B * pc.L * pc.R * pc.G * pc.N;
	uint angleCount = pc.B * pc.L * pc.A;
	uint biasCount = pc.H * pc.R * pc.N;
	uint mimoCount = pc.H * pc.R * pc.P;
	uint normCount = pc.H * pc.P;
	uint cursor = 0u;
	if (index < groupCount) {
		uint n = index % pc.N;
		uint q = index / pc.N;
		uint rg = q % (pc.R * pc.G);
		uint token = q / (pc.R * pc.G);
		uint r = rg / pc.G;
		uint g = rg % pc.G;
		uint headsPerGroup = pc.H / pc.G;
		float dc = 0.0f, db = 0.0f;
		for (uint local = 0u; local < headsPerGroup; ++local) {
			uint h = g * headsPerGroup + local;
			uint headIndex = ((token * pc.H + h) * pc.R + r) * pc.N + n;
			dc += oaLoadF32(heap[pc.dc_head_idx], headIndex);
			db += oaLoadF32(heap[pc.db_head_idx], headIndex);
		}
		oaStoreF32(heap[pc.dc_idx], index, dc);
		oaStoreF32(heap[pc.db_idx], index, db);
		return;
	}
	cursor += groupCount;
	if (index < cursor + angleCount) {
		uint localIndex = index - cursor;
		uint a = localIndex % pc.A;
		uint token = localIndex / pc.A;
		uint t = token % pc.L;
		uint b = token / pc.L;
		float value = 0.0f;
		for (uint h = 0u; h < pc.H; ++h)
			value += oaLoadF32(heap[pc.dangle_head_idx], ((b * pc.H + h) * pc.L + t) * pc.A + a);
		oaStoreF32(heap[pc.dangle_idx], localIndex, value);
		return;
	}
	cursor += angleCount;
	if (index < cursor + biasCount) {
		uint localIndex = index - cursor;
		float dc = 0.0f, db = 0.0f;
		for (uint b = 0u; b < pc.B; ++b) {
			uint partial = b * biasCount + localIndex;
			dc += oaLoadF32(heap[pc.dc_bias_batch_idx], partial);
			db += oaLoadF32(heap[pc.db_bias_batch_idx], partial);
		}
		oaStoreF32(heap[pc.dc_bias_idx], localIndex, dc);
		oaStoreF32(heap[pc.db_bias_idx], localIndex, db);
		return;
	}
	cursor += biasCount;
	if (index < cursor + pc.H) {
		uint h = index - cursor;
		float value = 0.0f;
		for (uint b = 0u; b < pc.B; ++b)
			value += oaLoadF32(heap[pc.dd_batch_idx], b * pc.H + h);
		oaStoreF32(heap[pc.dd_idx], h, value);
		return;
	}
	cursor += pc.H;
	if (index < cursor + mimoCount) {
		uint localIndex = index - cursor;
		float dx = 0.0f, dz = 0.0f, dout = 0.0f;
		for (uint b = 0u; b < pc.B; ++b) {
			uint partial = b * mimoCount + localIndex;
			dx += oaLoadF32(heap[pc.dmx_batch_idx], partial);
			dz += oaLoadF32(heap[pc.dmz_batch_idx], partial);
			dout += oaLoadF32(heap[pc.dmo_batch_idx], partial);
		}
		oaStoreF32(heap[pc.dmx_idx], localIndex, dx);
		oaStoreF32(heap[pc.dmz_idx], localIndex, dz);
		oaStoreF32(heap[pc.dmo_idx], localIndex, dout);
		return;
	}
	cursor += mimoCount;
	uint normIndex = index - cursor;
	if (normIndex < normCount) {
		float value = 0.0f;
		for (uint b = 0u; b < pc.B; ++b)
			value += oaLoadF32(heap[pc.dnorm_batch_idx], b * normCount + normIndex);
		oaStoreF32(heap[pc.dnorm_idx], normIndex, value);
	}
}
