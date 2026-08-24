// Short Mamba-3 backward, stage 2: forward state replay and gate derivative.
//
// One workgroup owns a (batch, head) scan and one lane owns a value row.  The
// recurrent state stays private while it is used to form the pre-gate output;
// only H[t-1], which the reverse scan needs, is written to global memory.  This
// avoids materializing the state and then reading the full history a second time.

import storage;
import activations;

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
	uint B;
	uint L;
	uint H;
	uint P;
	uint N;
	uint has_z;
	uint has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	uint p = threadId.x;
	if (bh >= pc.B * pc.H || p >= pc.P) return;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	float state[MAXN];
	for (uint n = 0u; n < pc.N; ++n) state[n] = 0.0f;
	float dh = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;

	for (uint t = 0u; t < pc.L; ++t) {
		uint tokBH = (b * pc.L + t) * pc.H + h;
		uint vec = tokBH * pc.P + p;
		uint stateBase = ((bh * pc.L + t) * pc.P + p) * pc.N;
		uint qkBase = tokBH * pc.N;
		float xv = oaLoadF32(heap[pc.x_idx], vec);
		float decay = oaLoadF32(heap[pc.decay_idx], tokBH);
		float scale = oaLoadF32(heap[pc.scale_idx], tokBH);
		float pre = (dh + oaLoadF32(heap[pc.gamma_idx], tokBH)
			* oaLoadF32(heap[pc.qk_idx], tokBH)) * xv;
		for (uint n = 0u; n < pc.N; ++n) {
			float previous = state[n];
			oaStoreF32(heap[pc.hprev_idx], stateBase + n, previous);
			pre += oaLoadF32(heap[pc.crot_idx], qkBase + n) * decay * previous;
			state[n] = decay * previous + scale * xv
				* oaLoadF32(heap[pc.brot_idx], qkBase + n);
		}

		float dy = oaLoadF32(heap[pc.dout_idx], vec);
		if (pc.has_z != 0u) {
			float zv = oaLoadF32(heap[pc.z_idx], vec);
			float sg = Sigmoid(zv);
			oaStoreF32(heap[pc.dpre_idx], vec, dy * zv * sg);
			float dsilu = sg + zv * sg * (1.0f - sg);
			oaStoreF32(heap[pc.dz_idx], vec, dy * pre * dsilu);
		} else {
			oaStoreF32(heap[pc.dpre_idx], vec, dy);
			oaStoreF32(heap[pc.dz_idx], vec, 0.0f);
		}
	}
}
