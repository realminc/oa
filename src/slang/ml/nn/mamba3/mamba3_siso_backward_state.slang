// Short Mamba-3 backward, stage 2: forward state replay and gate derivative.
//
// One workgroup owns a (batch, head) scan and one lane owns a value row.  The
// recurrent state stays private while it is used to form the pre-gate output;
// only H[t-1], which the reverse scan needs, is written to global memory.  This
// avoids materializing the state and then reading the full history a second time.

import storage;
import activations;
import attributes;

static const uint THREADS = 16u;
static const uint MAXN = 32u;

struct PushConstants {
	uint brot_idx;
	uint x_idx;
	uint decay_idx;
	uint scale_idx;
	uint crot_idx;
	uint z_idx;
	uint gamma_idx;
	uint d_idx;
	uint qk_idx;
	uint dout_idx;
	uint hprev_idx;
	uint dpre_idx;
	uint dz_idx;
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint has_z;
	uint has_d;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_backward_state")]
[domain("ml")]
[variant("short_state")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	uint p = threadId.x;
	if (bh >= push.b * push.h || p >= push.p) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	float state[MAXN];
	for (uint n = 0u; n < push.n; ++n) state[n] = 0.0f;
	float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;

	for (uint t = 0u; t < push.l; ++t) {
		uint tokBH = (b * push.l + t) * push.h + h;
		uint vec = tokBH * push.p + p;
		uint stateBase = ((bh * push.l + t) * push.p + p) * push.n;
		uint qkBase = tokBH * push.n;
		float xv = load_f32(storage_buffers[push.x_idx], vec);
		float decay = load_f32(storage_buffers[push.decay_idx], tokBH);
		float scale = load_f32(storage_buffers[push.scale_idx], tokBH);
		float pre = (dh + load_f32(storage_buffers[push.gamma_idx], tokBH)
			* load_f32(storage_buffers[push.qk_idx], tokBH)) * xv;
		for (uint n = 0u; n < push.n; ++n) {
			float previous = state[n];
			store_f32(storage_buffers[push.hprev_idx], stateBase + n, previous);
			pre += load_f32(storage_buffers[push.crot_idx], qkBase + n) * decay * previous;
			state[n] = decay * previous + scale * xv
				* load_f32(storage_buffers[push.brot_idx], qkBase + n);
		}

		float dy = load_f32(storage_buffers[push.dout_idx], vec);
		if (push.has_z != 0u) {
			float zv = load_f32(storage_buffers[push.z_idx], vec);
			float sg = sigmoid(zv);
			store_f32(storage_buffers[push.dpre_idx], vec, dy * zv * sg);
			float dsilu = sg + zv * sg * (1.0f - sg);
			store_f32(storage_buffers[push.dz_idx], vec, dy * pre * dsilu);
		} else {
			store_f32(storage_buffers[push.dpre_idx], vec, dy);
			store_f32(storage_buffers[push.dz_idx], vec, 0.0f);
		}
	}
}





