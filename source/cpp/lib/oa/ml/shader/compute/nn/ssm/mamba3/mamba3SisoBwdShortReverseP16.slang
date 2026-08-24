// Short Mamba-3 backward reverse scan specialized for P<=16.
//
// One lane owns one value row. Compared with the general 32-lane schedule this
// removes idle lanes at the NLP-suite P=16 corner and eliminates the per-row
// C/B partial-reduction tables. Internal lowering selects this shader only when
// the discovered hardware subgroup is at least 16 lanes, so every WaveActiveSum
// spans the complete workgroup. Lanes cover up to two N<=32 columns when they
// publish the row reductions.

import storage;

static const uint THREADS = 16u;
static const uint MAXN = 32u;

struct PushConstants {
	uint dpre_idx;
	uint crot_idx;
	uint brot_idx;
	uint theta_idx;
	uint gamma_idx;
	uint scale_idx;
	uint decay_idx;
	uint hprev_idx;
	uint x_idx;
	uint d_idx;
	uint qk_idx;
	uint dc_idx;
	uint db_idx;
	uint dx_idx;
	uint dadt_idx;
	uint dgamma_idx;
	uint dscale_idx;
	uint dtheta_idx;
	uint dcbias_idx;
	uint dbbias_idx;
	uint dd_idx;
	uint B;
	uint L;
	uint H;
	uint P;
	uint N;
	uint A;
	uint has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gsC[MAXN];
groupshared float gsB[MAXN];
groupshared float gsDCBias[MAXN];
groupshared float gsDBBias[MAXN];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	uint p = threadId.x;
	if (bh >= pc.B * pc.H) return;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	bool activeP = p < pc.P;
	float dState[MAXN];
	if (activeP) {
		for (uint n = 0u; n < pc.N; ++n) dState[n] = 0.0f;
	}
	for (uint n = threadId.x; n < pc.N; n += THREADS) {
		gsDCBias[n] = 0.0f;
		gsDBBias[n] = 0.0f;
	}
	float dDSum = 0.0f;
	float dh = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint rr = 0u; rr < pc.L; ++rr) {
		uint t = pc.L - 1u - rr;
		uint tokBH = (b * pc.L + t) * pc.H + h;
		uint qkBase = tokBH * pc.N;
		uint vecBase = tokBH * pc.P;
		for (uint n = threadId.x; n < pc.N; n += THREADS) {
			gsC[n] = oaLoadF32(heap[pc.crot_idx], qkBase + n);
			gsB[n] = oaLoadF32(heap[pc.brot_idx], qkBase + n);
		}
		GroupMemoryBarrierWithGroupSync();

		float xv = activeP ? oaLoadF32(heap[pc.x_idx], vecBase + p) : 0.0f;
		float dpre = activeP ? oaLoadF32(heap[pc.dpre_idx], vecBase + p) : 0.0f;
		float decay = oaLoadF32(heap[pc.decay_idx], tokBH);
		float scale = oaLoadF32(heap[pc.scale_idx], tokBH);
		float sumB = 0.0f;
		float partA = 0.0f;
		float partScale = 0.0f;
		float dcEven = 0.0f;
		float dcOdd = 0.0f;
		float dbEven = 0.0f;
		float dbOdd = 0.0f;
		for (uint n = 0u; n < pc.N; ++n) {
			float dcLocal = 0.0f;
			float dbLocal = 0.0f;
			if (activeP) {
				uint stateIndex = ((bh * pc.L + t) * pc.P + p) * pc.N + n;
				float previous = oaLoadF32(heap[pc.hprev_idx], stateIndex);
				float incoming = dState[n];
				float dhd = dpre * gsC[n] + incoming;
				sumB += incoming * gsB[n];
				partA += dhd * previous;
				partScale += incoming * xv * gsB[n];
				dcLocal = dpre * decay * previous;
				dbLocal = incoming * xv;
				dState[n] = decay * dhd;
			}
			float dc = WaveActiveSum(dcLocal);
			float db = WaveActiveSum(dbLocal);
			if (n == 2u * p) {
				dcEven = dc;
				dbEven = db;
			} else if (n == 2u * p + 1u) {
				dcOdd = dc;
				dbOdd = db;
			}
		}
		float totalDGQ = WaveActiveSum(activeP ? dpre * xv : 0.0f);
		float totalDA = WaveActiveSum(partA);
		float totalDScale = WaveActiveSum(partScale);

		float dqk = totalDGQ * oaLoadF32(heap[pc.gamma_idx], tokBH);
		if (p < pc.A) {
			uint i0 = 2u * p;
			uint i1 = i0 + 1u;
			float dcr0 = dcEven + dqk * gsB[i0];
			float dcr1 = dcOdd + dqk * gsB[i1];
			float dbr0 = scale * dbEven + dqk * gsC[i0];
			float dbr1 = scale * dbOdd + dqk * gsC[i1];
			float th = oaLoadF32(heap[pc.theta_idx], (bh * pc.L + t) * pc.A + p);
			float cs = cos(th);
			float sn = sin(th);
			float dc0 = dcr0 * cs + dcr1 * sn;
			float dc1 = -dcr0 * sn + dcr1 * cs;
			float db0 = dbr0 * cs + dbr1 * sn;
			float db1 = -dbr0 * sn + dbr1 * cs;
			oaStoreF32(heap[pc.dc_idx], qkBase + i0, dc0);
			oaStoreF32(heap[pc.dc_idx], qkBase + i1, dc1);
			oaStoreF32(heap[pc.db_idx], qkBase + i0, db0);
			oaStoreF32(heap[pc.db_idx], qkBase + i1, db1);
			gsDCBias[i0] += dc0;
			gsDCBias[i1] += dc1;
			gsDBBias[i0] += db0;
			gsDBBias[i1] += db1;
			oaStoreF32(heap[pc.dtheta_idx], (bh * pc.L + t) * pc.A + p,
				-dcr0 * gsC[i1] + dcr1 * gsC[i0] - dbr0 * gsB[i1] + dbr1 * gsB[i0]);
		}
		uint i0 = 2u * p;
		uint i1 = i0 + 1u;
		if (i0 >= 2u * pc.A && i0 < pc.N) {
			float dc = dcEven + dqk * gsB[i0];
			float db = scale * dbEven + dqk * gsC[i0];
			oaStoreF32(heap[pc.dc_idx], qkBase + i0, dc);
			oaStoreF32(heap[pc.db_idx], qkBase + i0, db);
			gsDCBias[i0] += dc;
			gsDBBias[i0] += db;
		}
		if (i1 >= 2u * pc.A && i1 < pc.N) {
			float dc = dcOdd + dqk * gsB[i1];
			float db = scale * dbOdd + dqk * gsC[i1];
			oaStoreF32(heap[pc.dc_idx], qkBase + i1, dc);
			oaStoreF32(heap[pc.db_idx], qkBase + i1, db);
			gsDCBias[i1] += dc;
			gsDBBias[i1] += db;
		}
		if (activeP) {
			oaStoreF32(heap[pc.dx_idx], vecBase + p,
				dpre * (dh + oaLoadF32(heap[pc.gamma_idx], tokBH)
					* oaLoadF32(heap[pc.qk_idx], tokBH)) + scale * sumB);
		}
		if (threadId.x == 0u) {
			oaStoreF32(heap[pc.dadt_idx], tokBH, totalDA * decay);
			oaStoreF32(heap[pc.dscale_idx], tokBH, totalDScale);
			oaStoreF32(heap[pc.dgamma_idx], tokBH,
				totalDGQ * oaLoadF32(heap[pc.qk_idx], tokBH) + totalDScale);
			if (pc.has_d != 0u) {
				dDSum += totalDGQ;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint n = threadId.x; n < pc.N; n += THREADS) {
		oaStoreF32(heap[pc.dcbias_idx], bh * pc.N + n, gsDCBias[n]);
		oaStoreF32(heap[pc.dbbias_idx], bh * pc.N + n, gsDBBias[n]);
	}
	if (threadId.x == 0u) oaStoreF32(heap[pc.dd_idx], bh, dDSum);
}
