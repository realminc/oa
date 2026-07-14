// EmpyrealmSisoBwd.slang — Empyrealm SISO selective state-space backward (bounded memory).
//
// Reverse-time companion to EmpyrealmSisoFwd.slang. One workgroup per (batch, head);
// each thread owns one V-channel row p of the outer-product state. The exact reverse
// recurrence is unchanged from the verified version; what changed is the STORAGE:
//
//   Phase A (forward recompute): store ONLY chunk-boundary state Hentry[BH,nchunks,P,N]
//     and chunk-entry cumulative angle ThetaEntry[BH,nchunks,A]. No per-token state.
//   Phase B (reverse over chunks): for each chunk (last→first), re-derive the chunk's
//     per-token states from its boundary into a reused buffer ChunkBuf[BH,CHUNK,P,N]
//     and its per-token angles into groupshared, then run the reverse scan over the
//     chunk. dH carry and the angle reverse-cumsum cross chunk boundaries naturally.
//
// Memory is O(nchunks + CHUNK) in the L dimension (≈ √L optimal) instead of O(L) —
// this is what removes the prior backward memory caveat. The per-token reverse math
// (rotary/trapezoidal/cumsum couplings, in-kernel shifted RMW) is identical to the
// gradchecked version, so the finite-difference gradcheck still validates it.
//
// Forward recap (post-bias/rotary C[t],B[t]; a=exp(ADT); qk=sum_n C*B):
//   Hd[p][n] = a[t]*H[t-1][p][n]
//   pre[t][p]= sum_n C[t][n]*Hd[p][n] + (D + gamma[t]*qk[t])*x[t][p]
//   y[t][p]  = pre[t][p]*silu(z[t][p])
//   H[t][p][n]= Hd[p][n] + scale[t]*x[t][p]*B[t][n]
//
// This is an exact copy of Mamba3SisoBwd.slang — Empyrealm branding for namespace separation.

import Storage;
import Activations;  // Sigmoid, Silu — shared module (was local dupes; byte-identical)

static const uint THREADS = 128u;  // >= headdim (P)
static const uint MAXN    = 128u;  // >= d_state (N)
static const uint MAXA    = 64u;   // >= num_rope_angles (A <= N/2)
static const uint CHUNK   = 32u;   // backward recompute granularity
static const uint MAX_WAVES = 32u; // THREADS / minimum supported subgroup width

struct PushConstants {
	uint c_idx;      // C_raw   [B,L,H,N]
	uint b_idx;      // B_raw   [B,L,H,N]
	uint x_idx;      // x = V   [B,L,H,P]
	uint z_idx;      // z       [B,L,H,P]
	uint adt_idx;    // ADT     [B,L,H]
	uint dt_idx;     // DT      [B,L,H]
	uint trap_idx;   // trap raw [B,L,H]
	uint angle_idx;  // angle raw [B,L,A]
	uint cbias_idx;  // [H,N]
	uint bbias_idx;  // [H,N]
	uint d_idx;      // [H]
	uint dout_idx;   // dY      [B,L,H,P]
	// Scratch (ReadWrite).
	uint entry_idx;     // Hentry      [BH,nchunks,P,N]  chunk-boundary state
	uint thetaentry_idx;// ThetaEntry  [BH,nchunks,A]    chunk-entry cumulative angle
	uint chunkbuf_idx;  // ChunkBuf    [BH,CHUNK,P,N]    reused per-chunk states
	// Outputs.
	uint dc_idx;     // dC_raw  [B,L,H,N]
	uint db_idx;     // dB_raw  [B,L,H,N]
	uint dx_idx;     // dx      [B,L,H,P]
	uint dz_idx;     // dz      [B,L,H,P]
	uint dadt_idx;   // dADT    [B,L,H]
	uint ddt_idx;    // dDt     [B,L,H]  (RMW)
	uint dtrap_idx;  // dTrap   [B,L,H]  (RMW)
	uint dangle_idx; // dAngle  [B,H,L,A]
	uint ddtok_idx;  // dD_tok  [B,L,H]
	// Config.
	uint B;
	uint L;
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
groupshared float gsThetaRun[MAXA];        // running cumulative angle (Phase A / recompute)
groupshared float gsThetaChunk[CHUNK][MAXA]; // per-token angle for the current chunk
groupshared float gsQK;
groupshared float gsDGQ;
groupshared float gsDa;
groupshared float gsDscale;
groupshared float gsDCrotSum[MAXN];
groupshared float gsDBrotSum[MAXN];
groupshared float gsDtheta[MAXA];
groupshared float gsS[MAXA];               // reverse-cumsum of dtheta (persists)
groupshared float gsDtTheta;
groupshared float gsDCWave[MAX_WAVES][MAXN];
groupshared float gsDBWave[MAX_WAVES][MAXN];
groupshared float gsScalarWave[3][MAX_WAVES];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h  = bh % pc.H;
	uint b  = bh / pc.H;
	uint p  = tid.x;
	bool actP = (p < pc.P);

	uint L = pc.L, H = pc.H, N = pc.N, A = pc.A, P = pc.P;
	uint numChunks = (L + CHUNK - 1u) / CHUNK;
	float Dh = (pc.has_d != 0u) ? OaLoadF32(heap[pc.d_idx], h) : 0.0f;

	// ───────────────────────── Phase A: store chunk boundaries ─────────────────────────
	// Replay the forward scan; store only the state + cumulative angle ENTERING each chunk.
	{
		float Hst[MAXN];
		for (uint n = 0u; n < N; n++) Hst[n] = 0.0f;
		for (uint k = tid.x; k < A; k += THREADS) gsThetaRun[k] = 0.0f;
		GroupMemoryBarrierWithGroupSync();

		for (uint t = 0u; t < L; t++) {
			uint tokBH  = (b * L + t) * H + h;
			uint vecBase = tokBH * P;
			uint qkBase  = tokBH * N;
			uint c = t / CHUNK;

			float dt   = OaLoadF32(heap[pc.dt_idx],  tokBH);
			float trap = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokBH));
			float a_t  = exp(OaLoadF32(heap[pc.adt_idx], tokBH));
			float shifted_gamma = 0.0f;
			if (t + 1u < L) {
				uint tN = (b * L + (t + 1u)) * H + h;
				shifted_gamma = OaLoadF32(heap[pc.dt_idx], tN) * (1.0f - Sigmoid(OaLoadF32(heap[pc.trap_idx], tN)));
			}
			float scale = dt * trap + shifted_gamma;

			// Store this chunk's entry state/angle the first time we reach its first token.
			if (t == c * CHUNK) {
				if (actP) {
					uint eBase = (((bh * numChunks + c) * P) + p) * N;
					for (uint n = 0u; n < N; n++) OaStoreF32(heap[pc.entry_idx], eBase + n, Hst[n]);
				}
				for (uint k = tid.x; k < A; k += THREADS)
					OaStoreF32(heap[pc.thetaentry_idx], (bh * numChunks + c) * A + k, gsThetaRun[k]);
			}

			// Advance angle, rotary B, advance state.
			uint angBase = (b * L + t) * A;
			for (uint k = tid.x; k < A; k += THREADS)
				gsThetaRun[k] += OaLoadF32(heap[pc.angle_idx], angBase + k) * dt;
			for (uint n = tid.x; n < N; n += THREADS)
				gsB[n] = OaLoadF32(heap[pc.b_idx], qkBase + n) + OaLoadF32(heap[pc.bbias_idx], h * N + n);
			GroupMemoryBarrierWithGroupSync();
			for (uint k = tid.x; k < A; k += THREADS) {
				float th = gsThetaRun[k];
				float cs = cos(th), sn = sin(th);
				uint i0 = 2u * k, i1 = i0 + 1u;
				float b0 = gsB[i0], b1 = gsB[i1];
				gsB[i0] = b0 * cs - b1 * sn;  gsB[i1] = b0 * sn + b1 * cs;
			}
			GroupMemoryBarrierWithGroupSync();
			if (actP) {
				float xv = OaLoadF32(heap[pc.x_idx], vecBase + p);
				for (uint n = 0u; n < N; n++) Hst[n] = a_t * Hst[n] + scale * xv * gsB[n];
			}
			GroupMemoryBarrierWithGroupSync();
		}
	}
	AllMemoryBarrierWithGroupSync();

	// ───────────────────────── Phase B: reverse over chunks ─────────────────────────
	float dH[MAXN];
	float Hprev[MAXN];
	for (uint n = 0u; n < N; n++) dH[n] = 0.0f;
	for (uint k = tid.x; k < A; k += THREADS) gsS[k] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint cc = 0u; cc < numChunks; cc++) {
		uint c = numChunks - 1u - cc;
		uint chunkStart = c * CHUNK;
		uint clen = min(CHUNK, L - chunkStart);

		// (1) Recompute this chunk's per-token states (ChunkBuf) and per-token angle.
		for (uint k = tid.x; k < A; k += THREADS)
			gsThetaRun[k] = OaLoadF32(heap[pc.thetaentry_idx], (bh * numChunks + c) * A + k);
		float Hcur[MAXN];
		if (actP) {
			uint eBase = (((bh * numChunks + c) * P) + p) * N;
			for (uint n = 0u; n < N; n++) Hcur[n] = OaLoadF32(heap[pc.entry_idx], eBase + n);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint lt = 0u; lt < clen; lt++) {
			uint t = chunkStart + lt;
			uint tokBH = (b * L + t) * H + h;
			float dt   = OaLoadF32(heap[pc.dt_idx], tokBH);
			float trap = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokBH));
			float a_t  = exp(OaLoadF32(heap[pc.adt_idx], tokBH));
			float shifted_gamma = 0.0f;
			if (t + 1u < L) {
				uint tN = (b * L + (t + 1u)) * H + h;
				shifted_gamma = OaLoadF32(heap[pc.dt_idx], tN) * (1.0f - Sigmoid(OaLoadF32(heap[pc.trap_idx], tN)));
			}
			float scale = dt * trap + shifted_gamma;
			uint angBase = (b * L + t) * A;
			for (uint k = tid.x; k < A; k += THREADS) {
				gsThetaRun[k] += OaLoadF32(heap[pc.angle_idx], angBase + k) * dt;
				gsThetaChunk[lt][k] = gsThetaRun[k];
			}
			for (uint n = tid.x; n < N; n += THREADS)
				gsB[n] = OaLoadF32(heap[pc.b_idx], tokBH * N + n) + OaLoadF32(heap[pc.bbias_idx], h * N + n);
			GroupMemoryBarrierWithGroupSync();
			for (uint k = tid.x; k < A; k += THREADS) {
				float th = gsThetaChunk[lt][k];
				float cs = cos(th), sn = sin(th);
				uint i0 = 2u * k, i1 = i0 + 1u;
				float b0 = gsB[i0], b1 = gsB[i1];
				gsB[i0] = b0 * cs - b1 * sn;  gsB[i1] = b0 * sn + b1 * cs;
			}
			GroupMemoryBarrierWithGroupSync();
			if (actP) {
				float xv = OaLoadF32(heap[pc.x_idx], tokBH * P + p);
				uint cbBase = (((bh * CHUNK + lt) * P) + p) * N;
				for (uint n = 0u; n < N; n++) {
					Hcur[n] = a_t * Hcur[n] + scale * xv * gsB[n];
					OaStoreF32(heap[pc.chunkbuf_idx], cbBase + n, Hcur[n]);
				}
			}
			GroupMemoryBarrierWithGroupSync();
		}

		// (2) Reverse scan over the chunk (token body identical to the verified kernel).
		for (uint ll = 0u; ll < clen; ll++) {
			uint lt = clen - 1u - ll;
			uint t  = chunkStart + lt;
			uint tokBH  = (b * L + t) * H + h;
			uint vecBase = tokBH * P;
			uint qkBase  = tokBH * N;

			float dt   = OaLoadF32(heap[pc.dt_idx],  tokBH);
			float trap = Sigmoid(OaLoadF32(heap[pc.trap_idx], tokBH));
			float a_t  = exp(OaLoadF32(heap[pc.adt_idx], tokBH));
			float shifted_gamma = 0.0f;
			if (t + 1u < L) {
				uint tN = (b * L + (t + 1u)) * H + h;
				shifted_gamma = OaLoadF32(heap[pc.dt_idx], tN) * (1.0f - Sigmoid(OaLoadF32(heap[pc.trap_idx], tN)));
			}
			float gamma = dt * trap;
			float scale = gamma + shifted_gamma;

			// Rebuild post-rotary C[t], B[t] from this chunk's stored angle.
			for (uint n = tid.x; n < N; n += THREADS) {
				gsC[n] = OaLoadF32(heap[pc.c_idx], qkBase + n) + OaLoadF32(heap[pc.cbias_idx], h * N + n);
				gsB[n] = OaLoadF32(heap[pc.b_idx], qkBase + n) + OaLoadF32(heap[pc.bbias_idx], h * N + n);
			}
			GroupMemoryBarrierWithGroupSync();
			for (uint k = tid.x; k < A; k += THREADS) {
				float th = gsThetaChunk[lt][k];
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
			if (tid.x == 0u) gsDtTheta = 0.0f;
			GroupMemoryBarrierWithGroupSync();
			float qk = gsQK;

			float xv = 0.0f, dpre = 0.0f, sumB = 0.0f;
			if (actP) {
				xv = OaLoadF32(heap[pc.x_idx], vecBase + p);
				// Hprev = H[t-1]: previous token in chunk, else chunk entry state.
				if (lt > 0u) {
					uint cbBase = (((bh * CHUNK + (lt - 1u)) * P) + p) * N;
					for (uint n = 0u; n < N; n++) Hprev[n] = OaLoadF32(heap[pc.chunkbuf_idx], cbBase + n);
				} else {
					uint eBase = (((bh * numChunks + c) * P) + p) * N;
					for (uint n = 0u; n < N; n++) Hprev[n] = OaLoadF32(heap[pc.entry_idx], eBase + n);
				}

				float pre = (Dh + gamma * qk) * xv;
				for (uint n = 0u; n < N; n++) pre += gsC[n] * (a_t * Hprev[n]);

				float dy = OaLoadF32(heap[pc.dout_idx], vecBase + p);
				if (pc.has_z != 0u) {
					float zv = OaLoadF32(heap[pc.z_idx], vecBase + p);
					float sg = Sigmoid(zv);
					dpre = dy * (zv * sg);
					float dsilu = sg + zv * sg * (1.0f - sg);
					OaStoreF32(heap[pc.dz_idx], vecBase + p, dy * pre * dsilu);
				} else {
					dpre = dy;
					OaStoreF32(heap[pc.dz_idx], vecBase + p, 0.0f);
				}
				for (uint n = 0u; n < N; n++) sumB += dH[n] * gsB[n];
			}

			// Reduce the P rows in parallel. The old implementation activated one p
			// thread at a time and issued P full-workgroup barriers per token.
			uint waveLane = WaveGetLaneIndex();
			uint waveSize = WaveGetLaneCount();
			uint waveId = tid.x / waveSize;
			uint waveCount = (THREADS + waveSize - 1u) / waveSize;
			float pDGQ = actP ? dpre * xv : 0.0f;
			float pDa = 0.0f, pDscale = 0.0f;
			for (uint n = 0u; n < N; n++) {
				float dcLocal = 0.0f, dbLocal = 0.0f;
				if (actP) {
					float hprev = Hprev[n];
					float hd = a_t * hprev;
					float dhd = dpre * gsC[n] + dH[n];
					pDa += dhd * hprev;
					pDscale += dH[n] * xv * gsB[n];
					dcLocal = dpre * hd;
					dbLocal = dH[n] * xv;
				}
				float dcWave = WaveActiveSum(dcLocal);
				float dbWave = WaveActiveSum(dbLocal);
				if (waveLane == 0u) {
					gsDCWave[waveId][n] = dcWave;
					gsDBWave[waveId][n] = dbWave;
				}
			}
			float dgqWave = WaveActiveSum(pDGQ);
			float daWave = WaveActiveSum(pDa);
			float dscaleWave = WaveActiveSum(pDscale);
			if (waveLane == 0u) {
				gsScalarWave[0][waveId] = dgqWave;
				gsScalarWave[1][waveId] = daWave;
				gsScalarWave[2][waveId] = dscaleWave;
			}
			GroupMemoryBarrierWithGroupSync();
			if (tid.x < N) {
				float dc = 0.0f, db = 0.0f;
				for (uint w = 0u; w < waveCount; w++) {
					dc += gsDCWave[w][tid.x];
					db += gsDBWave[w][tid.x];
				}
				gsDCrotSum[tid.x] = dc;
				gsDBrotSum[tid.x] = db;
			}
			if (tid.x == 0u) {
				gsDGQ = 0.0f; gsDa = 0.0f; gsDscale = 0.0f;
				for (uint w = 0u; w < waveCount; w++) {
					gsDGQ += gsScalarWave[0][w];
					gsDa += gsScalarWave[1][w];
					gsDscale += gsScalarWave[2][w];
				}
			}
			GroupMemoryBarrierWithGroupSync();

			float dGQ = gsDGQ;
			float dqk = dGQ * gamma;
			float dgam_pre = dGQ * qk;

			for (uint k = tid.x; k < A; k += THREADS) {
				uint i0 = 2u * k, i1 = i0 + 1u;
				float dCr0 = gsDCrotSum[i0] + dqk * gsB[i0];
				float dCr1 = gsDCrotSum[i1] + dqk * gsB[i1];
				float dBr0 = scale * gsDBrotSum[i0] + dqk * gsC[i0];
				float dBr1 = scale * gsDBrotSum[i1] + dqk * gsC[i1];
				float th = gsThetaChunk[lt][k];
				float cs = cos(th), sn = sin(th);
				OaStoreF32(heap[pc.dc_idx], qkBase + i0, dCr0 * cs + dCr1 * sn);
				OaStoreF32(heap[pc.dc_idx], qkBase + i1, -dCr0 * sn + dCr1 * cs);
				OaStoreF32(heap[pc.db_idx], qkBase + i0, dBr0 * cs + dBr1 * sn);
				OaStoreF32(heap[pc.db_idx], qkBase + i1, -dBr0 * sn + dBr1 * cs);
				gsDtheta[k] = -dCr0 * gsC[i1] + dCr1 * gsC[i0]
				            + -dBr0 * gsB[i1] + dBr1 * gsB[i0];
			}
			for (uint n = 2u * A + tid.x; n < N; n += THREADS) {
				OaStoreF32(heap[pc.dc_idx], qkBase + n, gsDCrotSum[n] + dqk * gsB[n]);
				OaStoreF32(heap[pc.db_idx], qkBase + n, scale * gsDBrotSum[n] + dqk * gsC[n]);
			}
			GroupMemoryBarrierWithGroupSync();

			for (uint k = tid.x; k < A; k += THREADS) {
				gsS[k] += gsDtheta[k];
				OaStoreF32(heap[pc.dangle_idx], (bh * L + t) * A + k, gsS[k] * dt);
			}
			GroupMemoryBarrierWithGroupSync();
			if (tid.x == 0u) {
				float acc = 0.0f;
				uint angBase = (b * L + t) * A;
				for (uint k = 0u; k < A; k++) acc += gsS[k] * OaLoadF32(heap[pc.angle_idx], angBase + k);
				gsDtTheta = acc;
			}

			if (actP) {
				float dx = dpre * (Dh + gamma * qk) + scale * sumB;
				OaStoreF32(heap[pc.dx_idx], vecBase + p, dx);
				for (uint n = 0u; n < N; n++) dH[n] = a_t * (dpre * gsC[n] + dH[n]);
			}

			if (tid.x == 0u) {
				OaStoreF32(heap[pc.dadt_idx], tokBH, gsDa * a_t);
				OaStoreF32(heap[pc.ddtok_idx], tokBH, dGQ);
				float dGamma_total = dgam_pre + gsDscale;
				float ddtBase = dGamma_total * trap + gsDtTheta;
				OaStoreF32(heap[pc.ddt_idx],   tokBH, OaLoadF32(heap[pc.ddt_idx],   tokBH) + ddtBase);
				OaStoreF32(heap[pc.dtrap_idx], tokBH, OaLoadF32(heap[pc.dtrap_idx], tokBH) + dGamma_total * dt);
				if (t + 1u < L) {
					uint tN = (b * L + (t + 1u)) * H + h;
					float dtN   = OaLoadF32(heap[pc.dt_idx], tN);
					float trapN = Sigmoid(OaLoadF32(heap[pc.trap_idx], tN));
					OaStoreF32(heap[pc.ddt_idx],   tN, OaLoadF32(heap[pc.ddt_idx],   tN) + gsDscale * (1.0f - trapN));
					OaStoreF32(heap[pc.dtrap_idx], tN, OaLoadF32(heap[pc.dtrap_idx], tN) + gsDscale * (-dtN));
				}
			}
			GroupMemoryBarrierWithGroupSync();
		}
	}
}
