// Mamba3SisoStep.slang — Mamba-3 SISO single-token recurrent step (autoregressive decode).
//
// One workgroup per (batch, head); thread p owns state row ssm_state[p][N]. State carried
// across steps: ssm_state[B,H,P,N], angle_state[B,H,A] (cumulative rotary angle),
// k_state[B,H,N] (previous token's post-rotary B), v_state[B,H,P] (previous token's x).
//
// The full scan's H[t] = a[t]H[t-1] + scale[t] x[t] B[t] with scale[t]=gamma[t]+dt[t+1](1-trap[t+1])
// needs the NEXT token's dt/trap. In streaming we defer that shifted term: at step t add the
// previous token's shifted contribution (using THIS token's dt,trap) to complete H[t-1], then
//   Hd      = a * ssm_state            (ssm_state now == complete H[t-1])
//   y[p]    = sum_n C[n]*Hd[n] + (D + gamma*qk)*x[p];  y[p] *= silu(z[p])
//   ssm_state[p][n] = Hd[n] + gamma*x[p]*B[n]          (gamma only; shifted deferred to t+1)
// then cache v_state=x, k_state=B(post-rotary), angle_state=Theta. Zero-init state ⇒ correct t=0.

import Storage;
import Activations;  // Sigmoid, Silu — shared module (was local dupes; byte-identical)

static const uint THREADS = 128u;
static const uint MAXN    = 128u;
static const uint MAXA    = 64u;

struct PushConstants {
	uint c_idx;      // C_raw [B,1,H,N]
	uint b_idx;      // B_raw [B,1,H,N]
	uint x_idx;      // x     [B,1,H,P]
	uint z_idx;      // z     [B,1,H,P]
	uint adt_idx;    // ADT   [B,1,H]
	uint dt_idx;     // DT    [B,1,H]
	uint trap_idx;   // trap raw [B,1,H]
	uint angle_idx;  // angle raw [B,1,A]
	uint cbias_idx;  // [H,N]
	uint bbias_idx;  // [H,N]
	uint d_idx;      // [H]
	uint y_idx;      // [B,1,H,P]
	uint ssm_idx;    // ssm_state   [B,H,P,N]  (RW)
	uint angst_idx;  // angle_state [B,H,A]    (RW)
	uint kst_idx;    // k_state     [B,H,N]    (RW)
	uint vst_idx;    // v_state     [B,H,P]    (RW)
	uint B;
	uint H;
	uint P;
	uint N;
	uint A;
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
	uint bh = gid.x;
	uint h  = bh % pc.H;
	uint b  = bh / pc.H;
	uint p  = tid.x;
	bool actP = (p < pc.P);

	uint H = pc.H, N = pc.N, A = pc.A, P = pc.P;
	float Dh = (pc.has_d != 0u) ? OaLoadF32(heap[pc.d_idx], h) : 0.0f;

	uint tokBH  = b * H + h;            // [B,1,H] scalar index
	uint vecBase = tokBH * P;
	uint qkBase  = tokBH * N;
	uint angBase = b * A;              // angle [B,1,A]
	uint ssmBase = (b * H + h) * P * N;
	uint kstBase = (b * H + h) * N;
	uint vstBase = (b * H + h) * P;
	uint angstBase = (b * H + h) * A;

	float dt   = OaLoadF32(heap[pc.dt_idx],  tokBH);
	float trap = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokBH));
	float a_t  = exp(OaLoadF32(heap[pc.adt_idx], tokBH));
	float gamma = dt * trap;
	float deferred = dt * (1.0f - trap);  // shifted_gamma[t-1] coefficient

	// Update cumulative rotary angle and build post-rotary C[t], B[t].
	for (uint k = tid.x; k < A; k += THREADS) {
		float th = OaLoadF32(heap[pc.angst_idx], angstBase + k) + OaLoadF32(heap[pc.angle_idx], angBase + k) * dt;
		gsTheta[k] = th;
		OaStoreF32(heap[pc.angst_idx], angstBase + k, th);  // persist for next step
	}
	for (uint n = tid.x; n < N; n += THREADS) {
		gsC[n] = OaLoadF32(heap[pc.c_idx], qkBase + n) + OaLoadF32(heap[pc.cbias_idx], h * N + n);
		gsB[n] = OaLoadF32(heap[pc.b_idx], qkBase + n) + OaLoadF32(heap[pc.bbias_idx], h * N + n);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint k = tid.x; k < A; k += THREADS) {
		float th = gsTheta[k];
		float cs = cos(th), sn = sin(th);
		uint i0 = 2u * k, i1 = i0 + 1u;
		float c0 = gsC[i0], c1 = gsC[i1];
		float b0 = gsB[i0], b1 = gsB[i1];
		gsC[i0] = c0 * cs - c1 * sn;  gsC[i1] = c0 * sn + c1 * cs;
		gsB[i0] = b0 * cs - b1 * sn;  gsB[i1] = b0 * sn + b1 * cs;
	}
	GroupMemoryBarrierWithGroupSync();
	if (tid.x == 0u) {
		float acc = 0.0f;
		for (uint n = 0u; n < N; n++) acc += gsC[n] * gsB[n];
		gsQK = acc;
	}
	GroupMemoryBarrierWithGroupSync();
	float qk = gsQK;

	if (actP) {
		float xv = OaLoadF32(heap[pc.x_idx], vecBase + p);
		float vprev = OaLoadF32(heap[pc.vst_idx], vstBase + p);  // x[t-1] for this row

		float y = 0.0f;
		for (uint n = 0u; n < N; n++) {
			// complete H[t-1]: add deferred shifted term v_state[p]*k_state[n]*dt*(1-trap)
			float s = OaLoadF32(heap[pc.ssm_idx], ssmBase + p * N + n)
			        + vprev * OaLoadF32(heap[pc.kst_idx], kstBase + n) * deferred;
			float hd = a_t * s;                  // decay to current position
			y += gsC[n] * hd;
			// store new state: Hd + gamma*x*B (shifted deferred to next step)
			OaStoreF32(heap[pc.ssm_idx], ssmBase + p * N + n, hd + gamma * xv * gsB[n]);
		}
		y += (Dh + gamma * qk) * xv;
		if (pc.has_z != 0u) y *= Silu(OaLoadF32(heap[pc.z_idx], vecBase + p));
		OaStoreF32(heap[pc.y_idx], vecBase + p, y);

		// cache this token's x for next step's deferred term.
		OaStoreF32(heap[pc.vst_idx], vstBase + p, xv);
	}
	// Barrier: all threads must finish reading the OLD k_state (in the deferred term)
	// before any thread overwrites it with this token's B.
	GroupMemoryBarrierWithGroupSync();
	for (uint n = tid.x; n < N; n += THREADS)
		OaStoreF32(heap[pc.kst_idx], kstBase + n, gsB[n]);
}
