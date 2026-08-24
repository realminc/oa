// Recompute each independent forward chunk and form the gate derivative.

import storage;
import activations;

static const uint THREADS = 64u;
static const uint CHUNK = 16u;
static const uint MAXN = 32u;

struct PushConstants {
	uint crot_idx;
	uint brot_idx;
	uint x_idx;
	uint z_idx;
	uint decay_idx;
	uint gamma_idx;
	uint scale_idx;
	uint qk_idx;
	uint d_idx;
	uint dout_idx;
	uint entry_idx;
	uint hprev_idx;
	uint dpre_idx;
	uint dz_idx;
	uint B;
	uint L;
	uint H;
	uint P;
	uint N;
	uint chunks;
	uint has_z;
	uint has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / pc.chunks;
	uint chunk = group - bh * pc.chunks;
	uint p = threadId.x;
	if (bh >= pc.B * pc.H || p >= pc.P) return;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, pc.L);
	float state[MAXN];
	uint entry = ((bh * pc.chunks + chunk) * pc.P + p) * pc.N;
	for (uint n = 0u; n < pc.N; ++n) {
		state[n] = oaLoadF32(heap[pc.entry_idx], entry + n);
	}
	float direct = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;

	for (uint t = begin; t < end; ++t) {
		uint scalar = (b * pc.L + t) * pc.H + h;
		uint vector = scalar * pc.P + p;
		uint stateBase = scalar * pc.N;
		float decay = oaLoadF32(heap[pc.decay_idx], scalar);
		float value = oaLoadF32(heap[pc.x_idx], vector);
		float pre = (direct + oaLoadF32(heap[pc.gamma_idx], scalar)
			* oaLoadF32(heap[pc.qk_idx], scalar)) * value;
		for (uint n = 0u; n < pc.N; ++n) {
			float previous = state[n];
			uint history = ((bh * pc.L + t) * pc.P + p) * pc.N + n;
			oaStoreF32(heap[pc.hprev_idx], history, previous);
			pre += oaLoadF32(heap[pc.crot_idx], stateBase + n) * decay * previous;
			state[n] = decay * previous + oaLoadF32(heap[pc.scale_idx], scalar) * value
				* oaLoadF32(heap[pc.brot_idx], stateBase + n);
		}
		float dy = oaLoadF32(heap[pc.dout_idx], vector);
		if (pc.has_z != 0u) {
			float gate = oaLoadF32(heap[pc.z_idx], vector);
			float sigmoid = Sigmoid(gate);
			oaStoreF32(heap[pc.dpre_idx], vector, dy * gate * sigmoid);
			oaStoreF32(heap[pc.dz_idx], vector,
				dy * pre * (sigmoid + gate * sigmoid * (1.0f - sigmoid)));
		} else {
			oaStoreF32(heap[pc.dpre_idx], vector, dy);
			oaStoreF32(heap[pc.dz_idx], vector, 0.0f);
		}
	}
}
