// Recompute each independent forward chunk and form the gate derivative.

import storage;
import activations;
import attributes;

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
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint chunks;
	uint has_z;
	uint has_d;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_chunk_dpre")]
[domain("ml")]
[variant("chunk16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / push.chunks;
	uint chunk = group - bh * push.chunks;
	uint p = threadId.x;
	if (bh >= push.b * push.h || p >= push.p) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, push.l);
	float state[MAXN];
	uint entry = ((bh * push.chunks + chunk) * push.p + p) * push.n;
	for (uint n = 0u; n < push.n; ++n) {
		state[n] = load_f32(storage_buffers[push.entry_idx], entry + n);
	}
	float direct = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;

	for (uint t = begin; t < end; ++t) {
		uint scalar = (b * push.l + t) * push.h + h;
		uint vector = scalar * push.p + p;
		uint stateBase = scalar * push.n;
		float decay = load_f32(storage_buffers[push.decay_idx], scalar);
		float value = load_f32(storage_buffers[push.x_idx], vector);
		float pre = (direct + load_f32(storage_buffers[push.gamma_idx], scalar)
			* load_f32(storage_buffers[push.qk_idx], scalar)) * value;
		for (uint n = 0u; n < push.n; ++n) {
			float previous = state[n];
			uint history = ((bh * push.l + t) * push.p + p) * push.n + n;
			store_f32(storage_buffers[push.hprev_idx], history, previous);
			pre += load_f32(storage_buffers[push.crot_idx], stateBase + n) * decay * previous;
			state[n] = decay * previous + load_f32(storage_buffers[push.scale_idx], scalar) * value
				* load_f32(storage_buffers[push.brot_idx], stateBase + n);
		}
		float dy = load_f32(storage_buffers[push.dout_idx], vector);
		if (push.has_z != 0u) {
			float gate = load_f32(storage_buffers[push.z_idx], vector);
			float gate_sigmoid = sigmoid(gate);
			store_f32(storage_buffers[push.dpre_idx], vector, dy * gate * gate_sigmoid);
			store_f32(storage_buffers[push.dz_idx], vector,
				dy * pre * (gate_sigmoid + gate * gate_sigmoid * (1.0f - gate_sigmoid)));
		} else {
			store_f32(storage_buffers[push.dpre_idx], vector, dy);
			store_f32(storage_buffers[push.dz_idx], vector, 0.0f);
		}
	}
}
