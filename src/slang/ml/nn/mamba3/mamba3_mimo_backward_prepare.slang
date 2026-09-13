// Recompute deterministic MIMO forward intermediates for the adjoint.
// This stage owns the O(B*H*L*P*N) history needed by the exact reverse scan;
// no full history is retained by the public forward operation.

import storage;
import activations;
import attributes;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;
static const uint MAXR = 8u;

struct PushConstants {
	uint c_idx, b_idx, x_idx, adt_idx, dt_idx, trap_idx, angle_idx;
	uint cbias_idx, bbias_idx, mx_idx;
	uint q_idx, k_idx, theta_idx, hprev_idx;
	uint decay_idx, gamma_idx, scale_idx;
	uint b, l, h, g, p, n, a, r;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float gsQ[MAXR * MAXN];
groupshared float gsK[MAXR * MAXN];
groupshared float gsTheta[MAXA];

[kernel_name("mamba3_mimo_backward_prepare")]
[domain("ml")]
[variant("prepare")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint g = h / (push.h / push.g);
	uint p = tid.x;
	bool activeP = p < push.p;
	float state[MAXN];
	for (uint n = 0u; n < push.n; ++n) state[n] = 0.0f;
	for (uint a = tid.x; a < push.a; a += THREADS) gsTheta[a] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint t = 0u; t < push.l; ++t) {
		uint scalar = (b * push.l + t) * push.h + h;
		float dt = load_f32(storage_buffers[push.dt_idx], scalar);
		float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], scalar));
		float decay = exp(load_f32(storage_buffers[push.adt_idx], scalar));
		float gamma = dt * trap;
		float shifted = 0.0f;
		if (t + 1u < push.l) {
			uint next = (b * push.l + t + 1u) * push.h + h;
			float nextTrap = sigmoid(load_f32(storage_buffers[push.trap_idx], next));
			shifted = load_f32(storage_buffers[push.dt_idx], next) * (1.0f - nextTrap);
		}
		store_f32(storage_buffers[push.decay_idx], scalar, decay);
		store_f32(storage_buffers[push.gamma_idx], scalar, gamma);
		store_f32(storage_buffers[push.scale_idx], scalar, gamma + shifted);
		uint angleBase = (b * push.l + t) * push.a;
		for (uint a = tid.x; a < push.a; a += THREADS) {
			gsTheta[a] += load_f32(storage_buffers[push.angle_idx], angleBase + a) * dt;
			uint thetaOut = ((b * push.h + h) * push.l + t) * push.a + a;
			store_f32(storage_buffers[push.theta_idx], thetaOut, gsTheta[a]);
		}
		for (uint linear = tid.x; linear < push.r * push.n; linear += THREADS) {
			uint r = linear / push.n;
			uint n = linear % push.n;
			uint rg = r * push.g + g;
			uint input = (((b * push.l + t) * (push.r * push.g) + rg) * push.n) + n;
			uint bias = (h * push.r + r) * push.n + n;
			gsQ[linear] = load_f32(storage_buffers[push.c_idx], input) + load_f32(storage_buffers[push.cbias_idx], bias);
			gsK[linear] = load_f32(storage_buffers[push.b_idx], input) + load_f32(storage_buffers[push.bbias_idx], bias);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint linear = tid.x; linear < push.r * push.a; linear += THREADS) {
			uint r = linear / push.a;
			uint a = linear % push.a;
			uint i0 = r * push.n + 2u * a;
			uint i1 = i0 + 1u;
			float cs = cos(gsTheta[a]), sn = sin(gsTheta[a]);
			float q0 = gsQ[i0], q1 = gsQ[i1];
			float k0 = gsK[i0], k1 = gsK[i1];
			gsQ[i0] = q0 * cs - q1 * sn;
			gsQ[i1] = q0 * sn + q1 * cs;
			gsK[i0] = k0 * cs - k1 * sn;
			gsK[i1] = k0 * sn + k1 * cs;
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint linear = tid.x; linear < push.r * push.n; linear += THREADS) {
			uint outIndex = ((((b * push.l + t) * push.h + h) * push.r) * push.n) + linear;
			store_f32(storage_buffers[push.q_idx], outIndex, gsQ[linear]);
			store_f32(storage_buffers[push.k_idx], outIndex, gsK[linear]);
		}
		if (activeP) {
			uint xp = ((b * push.l + t) * push.h + h) * push.p + p;
			float xv = load_f32(storage_buffers[push.x_idx], xp);
			uint history = (((b * push.h + h) * push.l + t) * push.p + p) * push.n;
			for (uint n = 0u; n < push.n; ++n) {
				store_f32(storage_buffers[push.hprev_idx], history + n, state[n]);
				float u = 0.0f;
				for (uint r = 0u; r < push.r; ++r) {
					uint rp = (h * push.r + r) * push.p + p;
					u += xv * load_f32(storage_buffers[push.mx_idx], rp) * gsK[r * push.n + n];
				}
				state[n] = decay * state[n]
					+ (gamma + shifted) * u;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
