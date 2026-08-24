// Complete independent-chunk Mamba-3 reverse pass.
//
// Forward H[t-1] history and reverse exit adjoints make the 16-token chunks
// independent. Each value-row lane keeps its state adjoint private while
// workgroup reductions combine value rows for C/B and scalar derivatives.

import storage;

static const uint THREADS = 64u;
static const uint CHUNK = 16u;
static const uint MAXN = 32u;
static const uint MAXA = 8u;

struct PushConstants {
	uint crot_idx;
	uint brot_idx;
	uint theta_idx;
	uint x_idx;
	uint decay_idx;
	uint gamma_idx;
	uint scale_idx;
	uint qk_idx;
	uint d_idx;
	uint dpre_idx;
	uint hprev_idx;
	uint exit_idx;
	uint dc_idx;
	uint db_idx;
	uint dx_idx;
	uint dadt_idx;
	uint dgamma_idx;
	uint dscale_idx;
	uint dtheta_idx;
	uint ddtoken_idx;
	uint B;
	uint L;
	uint H;
	uint P;
	uint N;
	uint A;
	uint chunks;
	uint has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gsC[MAXN];
groupshared float gsB[MAXN];
groupshared float gsDCPart[THREADS][MAXN];
groupshared float gsDBPart[THREADS][MAXN];
groupshared float gsDGQPart[THREADS];
groupshared float gsDAdtPart[THREADS];
groupshared float gsDScalePart[THREADS];
groupshared float gsDCSum[MAXN];
groupshared float gsDBSum[MAXN];
groupshared float gsDGQ;
groupshared float gsDAdt;
groupshared float gsDScale;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / pc.chunks;
	uint chunk = group - bh * pc.chunks;
	uint p = threadId.x;
	if (bh >= pc.B * pc.H) return;
	bool active = p < pc.P;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, pc.L);
	uint chunkState = ((bh * pc.chunks + chunk) * pc.P + p) * pc.N;
	float dState[MAXN];
	if (active) {
		for (uint n = 0u; n < pc.N; ++n) {
			dState[n] = oaLoadF32(heap[pc.exit_idx], chunkState + n);
		}
	}
	float direct = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;

	for (uint rr = 0u; rr < end - begin; ++rr) {
		uint t = end - 1u - rr;
		uint scalar = (b * pc.L + t) * pc.H + h;
		uint vector = scalar * pc.P + p;
		uint stateBase = scalar * pc.N;
		float hPrev[MAXN];
		if (active) {
			for (uint n = 0u; n < pc.N; ++n) {
				uint history = ((bh * pc.L + t) * pc.P + p) * pc.N + n;
				hPrev[n] = oaLoadF32(heap[pc.hprev_idx], history);
			}
		}

		for (uint n = threadId.x; n < pc.N; n += THREADS) {
			gsC[n] = oaLoadF32(heap[pc.crot_idx], stateBase + n);
			gsB[n] = oaLoadF32(heap[pc.brot_idx], stateBase + n);
		}
		GroupMemoryBarrierWithGroupSync();
		float x = active ? oaLoadF32(heap[pc.x_idx], vector) : 0.0f;
		float dpre = active ? oaLoadF32(heap[pc.dpre_idx], vector) : 0.0f;
		float decay = oaLoadF32(heap[pc.decay_idx], scalar);
		float localDAdt = 0.0f;
		float localDScale = 0.0f;
		float sumB = 0.0f;
		for (uint n = 0u; n < pc.N; ++n) {
			float dc = 0.0f;
			float db = 0.0f;
			if (active) {
				float incoming = dState[n];
				float dDecayed = dpre * gsC[n] + incoming;
				localDAdt += dDecayed * hPrev[n];
				localDScale += incoming * x * gsB[n];
				sumB += incoming * gsB[n];
				dc = dpre * decay * hPrev[n];
				db = incoming * x;
			}
			gsDCPart[p][n] = dc;
			gsDBPart[p][n] = db;
		}
		gsDGQPart[p] = active ? dpre * x : 0.0f;
		gsDAdtPart[p] = localDAdt;
		gsDScalePart[p] = localDScale;
		GroupMemoryBarrierWithGroupSync();

		if (threadId.x < pc.N) {
			float dc = 0.0f;
			float db = 0.0f;
			for (uint row = 0u; row < pc.P; ++row) {
				dc += gsDCPart[row][threadId.x];
				db += gsDBPart[row][threadId.x];
			}
			gsDCSum[threadId.x] = dc;
			gsDBSum[threadId.x] = db;
		}
		if (threadId.x == 0u) {
			gsDGQ = 0.0f;
			gsDAdt = 0.0f;
			gsDScale = 0.0f;
			for (uint row = 0u; row < pc.P; ++row) {
				gsDGQ += gsDGQPart[row];
				gsDAdt += gsDAdtPart[row];
				gsDScale += gsDScalePart[row];
			}
		}
		GroupMemoryBarrierWithGroupSync();

		float dqk = gsDGQ * oaLoadF32(heap[pc.gamma_idx], scalar);
		for (uint k = threadId.x; k < pc.A; k += THREADS) {
			uint i0 = 2u * k;
			uint i1 = i0 + 1u;
			float dcr0 = gsDCSum[i0] + dqk * gsB[i0];
			float dcr1 = gsDCSum[i1] + dqk * gsB[i1];
			float dbr0 = oaLoadF32(heap[pc.scale_idx], scalar) * gsDBSum[i0]
				+ dqk * gsC[i0];
			float dbr1 = oaLoadF32(heap[pc.scale_idx], scalar) * gsDBSum[i1]
				+ dqk * gsC[i1];
			float theta = oaLoadF32(heap[pc.theta_idx], (bh * pc.L + t) * pc.A + k);
			float cs = cos(theta);
			float sn = sin(theta);
			float dc0 = dcr0 * cs + dcr1 * sn;
			float dc1 = -dcr0 * sn + dcr1 * cs;
			float db0 = dbr0 * cs + dbr1 * sn;
			float db1 = -dbr0 * sn + dbr1 * cs;
			oaStoreF32(heap[pc.dc_idx], stateBase + i0, dc0);
			oaStoreF32(heap[pc.dc_idx], stateBase + i1, dc1);
			oaStoreF32(heap[pc.db_idx], stateBase + i0, db0);
			oaStoreF32(heap[pc.db_idx], stateBase + i1, db1);
			oaStoreF32(heap[pc.dtheta_idx], (bh * pc.L + t) * pc.A + k,
				-dcr0 * gsC[i1] + dcr1 * gsC[i0]
				-dbr0 * gsB[i1] + dbr1 * gsB[i0]);
		}
		for (uint n = 2u * pc.A + threadId.x; n < pc.N; n += THREADS) {
			float dc = gsDCSum[n] + dqk * gsB[n];
			float db = oaLoadF32(heap[pc.scale_idx], scalar) * gsDBSum[n]
				+ dqk * gsC[n];
			oaStoreF32(heap[pc.dc_idx], stateBase + n, dc);
			oaStoreF32(heap[pc.db_idx], stateBase + n, db);
		}
		if (active) {
			oaStoreF32(heap[pc.dx_idx], vector,
				dpre * (direct + oaLoadF32(heap[pc.gamma_idx], scalar)
					* oaLoadF32(heap[pc.qk_idx], scalar))
				+ oaLoadF32(heap[pc.scale_idx], scalar) * sumB);
			for (uint n = 0u; n < pc.N; ++n) {
				dState[n] = decay * (dpre * gsC[n] + dState[n]);
			}
		}
		if (threadId.x == 0u) {
			oaStoreF32(heap[pc.dadt_idx], scalar, gsDAdt * decay);
			oaStoreF32(heap[pc.dgamma_idx], scalar,
				gsDGQ * oaLoadF32(heap[pc.qk_idx], scalar) + gsDScale);
			oaStoreF32(heap[pc.dscale_idx], scalar, gsDScale);
			oaStoreF32(heap[pc.ddtoken_idx], scalar,
				pc.has_d != 0u ? gsDGQ : 0.0f);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
