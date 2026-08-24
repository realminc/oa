// Recompute deterministic MIMO forward intermediates for the adjoint.
// This stage owns the O(B*H*L*P*N) history needed by the exact reverse scan;
// no full history is retained by the public forward operation.

import storage;
import activations;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;
static const uint MAXR = 8u;

struct PushConstants {
	uint c_idx, b_idx, x_idx, adt_idx, dt_idx, trap_idx, angle_idx;
	uint cbias_idx, bbias_idx, mx_idx;
	uint q_idx, k_idx, theta_idx, hprev_idx;
	uint decay_idx, gamma_idx, scale_idx;
	uint B, L, H, G, P, N, A, R;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gsQ[MAXR * MAXN];
groupshared float gsK[MAXR * MAXN];
groupshared float gsTheta[MAXA];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint g = h / (pc.H / pc.G);
	uint p = tid.x;
	bool activeP = p < pc.P;
	float state[MAXN];
	for (uint n = 0u; n < pc.N; ++n) state[n] = 0.0f;
	for (uint a = tid.x; a < pc.A; a += THREADS) gsTheta[a] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint t = 0u; t < pc.L; ++t) {
		uint scalar = (b * pc.L + t) * pc.H + h;
		float dt = oaLoadF32(heap[pc.dt_idx], scalar);
		float trap = Sigmoid(oaLoadF32(heap[pc.trap_idx], scalar));
		float decay = exp(oaLoadF32(heap[pc.adt_idx], scalar));
		float gamma = dt * trap;
		float shifted = 0.0f;
		if (t + 1u < pc.L) {
			uint next = (b * pc.L + t + 1u) * pc.H + h;
			float nextTrap = Sigmoid(oaLoadF32(heap[pc.trap_idx], next));
			shifted = oaLoadF32(heap[pc.dt_idx], next) * (1.0f - nextTrap);
		}
		oaStoreF32(heap[pc.decay_idx], scalar, decay);
		oaStoreF32(heap[pc.gamma_idx], scalar, gamma);
		oaStoreF32(heap[pc.scale_idx], scalar, gamma + shifted);
		uint angleBase = (b * pc.L + t) * pc.A;
		for (uint a = tid.x; a < pc.A; a += THREADS) {
			gsTheta[a] += oaLoadF32(heap[pc.angle_idx], angleBase + a) * dt;
			uint thetaOut = ((b * pc.H + h) * pc.L + t) * pc.A + a;
			oaStoreF32(heap[pc.theta_idx], thetaOut, gsTheta[a]);
		}
		for (uint linear = tid.x; linear < pc.R * pc.N; linear += THREADS) {
			uint r = linear / pc.N;
			uint n = linear % pc.N;
			uint rg = r * pc.G + g;
			uint input = (((b * pc.L + t) * (pc.R * pc.G) + rg) * pc.N) + n;
			uint bias = (h * pc.R + r) * pc.N + n;
			gsQ[linear] = oaLoadF32(heap[pc.c_idx], input) + oaLoadF32(heap[pc.cbias_idx], bias);
			gsK[linear] = oaLoadF32(heap[pc.b_idx], input) + oaLoadF32(heap[pc.bbias_idx], bias);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint linear = tid.x; linear < pc.R * pc.A; linear += THREADS) {
			uint r = linear / pc.A;
			uint a = linear % pc.A;
			uint i0 = r * pc.N + 2u * a;
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
		for (uint linear = tid.x; linear < pc.R * pc.N; linear += THREADS) {
			uint outIndex = ((((b * pc.L + t) * pc.H + h) * pc.R) * pc.N) + linear;
			oaStoreF32(heap[pc.q_idx], outIndex, gsQ[linear]);
			oaStoreF32(heap[pc.k_idx], outIndex, gsK[linear]);
		}
		if (activeP) {
			uint xp = ((b * pc.L + t) * pc.H + h) * pc.P + p;
			float xv = oaLoadF32(heap[pc.x_idx], xp);
			uint history = (((b * pc.H + h) * pc.L + t) * pc.P + p) * pc.N;
			for (uint n = 0u; n < pc.N; ++n) {
				oaStoreF32(heap[pc.hprev_idx], history + n, state[n]);
				float u = 0.0f;
				for (uint r = 0u; r < pc.R; ++r) {
					uint rp = (h * pc.R + r) * pc.P + p;
					u += xv * oaLoadF32(heap[pc.mx_idx], rp) * gsK[r * pc.N + n];
				}
				state[n] = decay * state[n]
					+ (gamma + shifted) * u;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
