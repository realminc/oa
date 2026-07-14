// Mamba3SisoFwd.slang — Mamba-3 SISO selective state-space forward (full-state, fused).
//
// One workgroup per (batch, head). Each thread owns one V-channel row p of the
// outer-product state Hst[p][N]. The sequence is scanned sequentially in time so
// the data-dependent decay (per-token ADT), the trapezoidal discretization, and
// the cumulative rotary angle can all be applied exactly, matching the reference
// official Mamba-3 kernel (mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py).
//
// Per token t (head h, batch b):
//   a_t        = exp(ADT_t)
//   C_t,B_t    = rotary(  RMSNorm'd C,B  + bias , Theta_t )      (Theta_t = cumsum angle*DT)
//   gamma_t    = DT_t * sigmoid(trap_t)
//   scale_t    = gamma_t + DT_{t+1} * (1 - sigmoid(trap_{t+1}))
//   qk_t       = sum_n C_t[n] * B_t[n]
//   Hd[n]      = a_t * Hst[n]                       (decay prior state)
//   y[p]       = sum_n C_t[n] * Hd[n]  +  (D_h + gamma_t*qk_t) * x_t[p]
//   y[p]      *= silu(z_t[p])                       (if has_z)
//   Hst[n]     = Hd[n] + scale_t * x_t[p] * B_t[n]  (add current token)
//
// Layouts (all FP32):
//   C,B      : [B, L, H, N]      x,z,y : [B, L, H, P]
//   ADT,DT   : [B, L, H]         trap  : [B, L, H]
//   angle    : [B, L, A]   (shared across heads)    C_bias,B_bias : [H, N]   D : [H]

import Storage;
import Activations;

static const uint THREADS = 128u;  // >= headdim (P)
static const uint MAXN    = 128u;  // >= d_state (N)
static const uint MAXA    = 64u;   // >= num_rope_angles (A <= N/2)

struct PushConstants {
	// Buffer heap indices (auto-filled by Stream.cpp, in buffer order).
	uint c_idx;
	uint b_idx;
	uint x_idx;
	uint z_idx;
	uint adt_idx;
	uint dt_idx;
	uint trap_idx;
	uint angle_idx;
	uint cbias_idx;
	uint bbias_idx;
	uint d_idx;
	uint y_idx;
	// Config.
	uint B;
	uint L;
	uint H;
	uint P;        // headdim (headdim_v)
	uint N;        // d_state (headdim_qk)
	uint A;        // num_rope_angles
	uint has_z;
	uint has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gsC[MAXN];
groupshared float gsB[MAXN];
groupshared float gsTheta[MAXA];
groupshared float gsQK;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh   = gid.x;
	uint h    = bh % pc.H;
	uint b    = bh / pc.H;
	uint p    = tid.x;                 // V-channel row this thread owns
	bool actP = (p < pc.P);

	uint L = pc.L;
	uint H = pc.H;
	uint N = pc.N;
	uint A = pc.A;

	float Dh = (pc.has_d != 0u) ? OaLoadF32(heap[pc.d_idx], h) : 0.0f;

	// Local per-thread state row Hst[0..N).
	float Hst[MAXN];
	for (uint n = 0u; n < N; n++) Hst[n] = 0.0f;

	// Running cumulative rotary angle Theta (persists across time).
	for (uint k = tid.x; k < A; k += THREADS) gsTheta[k] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint t = 0u; t < L; t++) {
		uint tokBH = (b * L + t) * H + h;            // index into [B,L,H] scalars
		uint vecBase = tokBH * pc.P;                  // index into x,z,y [B,L,H,P]
		uint qkBase  = tokBH * N;                     // index into C,B   [B,L,H,N]

		float dt   = OaLoadF32(heap[pc.dt_idx],  tokBH);
		float adt  = OaLoadF32(heap[pc.adt_idx], tokBH);
		float trap = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokBH));
		float a_t  = exp(adt);

		// shifted_gamma uses the NEXT token (0 at the last token).
		float shifted_gamma = 0.0f;
		if (t + 1u < L) {
			uint tokNext = (b * L + (t + 1u)) * H + h;
			float dtN   = OaLoadF32(heap[pc.dt_idx],   tokNext);
			float trapN = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokNext));
			shifted_gamma = dtN * (1.0f - trapN);
		}
		float gamma = dt * trap;
		float scale = gamma + shifted_gamma;

		// Load C,B (already RMSNorm'd) and add per-head bias.
		for (uint n = tid.x; n < N; n += THREADS) {
			gsC[n] = OaLoadF32(heap[pc.c_idx], qkBase + n) + OaLoadF32(heap[pc.cbias_idx], h * N + n);
			gsB[n] = OaLoadF32(heap[pc.b_idx], qkBase + n) + OaLoadF32(heap[pc.bbias_idx], h * N + n);
		}
		// Advance cumulative angle by angle_raw_t * dt (inclusive of current token).
		uint angBase = (b * L + t) * A;
		for (uint k = tid.x; k < A; k += THREADS) {
			gsTheta[k] += OaLoadF32(heap[pc.angle_idx], angBase + k) * dt;
		}
		GroupMemoryBarrierWithGroupSync();

		// Interleaved rotary on the first A pairs of the N axis.
		for (uint k = tid.x; k < A; k += THREADS) {
			float th = gsTheta[k];
			float cs = cos(th);
			float sn = sin(th);
			uint i0 = 2u * k;
			uint i1 = i0 + 1u;
			float c0 = gsC[i0], c1 = gsC[i1];
			float b0 = gsB[i0], b1 = gsB[i1];
			gsC[i0] = c0 * cs - c1 * sn;  gsC[i1] = c0 * sn + c1 * cs;
			gsB[i0] = b0 * cs - b1 * sn;  gsB[i1] = b0 * sn + b1 * cs;
		}
		GroupMemoryBarrierWithGroupSync();

		// qk_t = sum_n C_t[n] * B_t[n]  (rotation-invariant; compute once).
		if (tid.x == 0u) {
			float acc = 0.0f;
			for (uint n = 0u; n < N; n++) acc += gsC[n] * gsB[n];
			gsQK = acc;
		}
		GroupMemoryBarrierWithGroupSync();
		float qk = gsQK;

		// Recurrence for this thread's state row.
		if (actP) {
			float xv = OaLoadF32(heap[pc.x_idx], vecBase + p);
			float y  = 0.0f;
			for (uint n = 0u; n < N; n++) {
				float hd = a_t * Hst[n];
				y += gsC[n] * hd;
				Hst[n] = hd + scale * xv * gsB[n];
			}
			y += (Dh + gamma * qk) * xv;
			if (pc.has_z != 0u) {
				y *= Silu(OaLoadF32(heap[pc.z_idx], vecBase + p));
			}
			OaStoreF32(heap[pc.y_idx], vecBase + p, y);
		}
		GroupMemoryBarrierWithGroupSync();  // protect gsC/gsB before next token
	}
}
