// Short Mamba-3 backward reverse scan specialized for P<=16.
//
// One lane owns one value row. Compared with the general 32-lane schedule this
// removes idle lanes at the NLP-suite P=16 corner. OARS uses explicit
// workgroup-shared reductions here, preserving the donor operation order without
// introducing a subgroup-size requirement. Lanes cover up to two N<=32 columns
// when they publish the row reductions.

import storage;
import attributes;

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
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint a;
	uint has_d;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float gsC[MAXN];
groupshared float gsB[MAXN];
groupshared float gsDCBias[MAXN];
groupshared float gsDBBias[MAXN];
groupshared float gsReduceA[THREADS];
groupshared float gsReduceB[THREADS];
groupshared float gsReduceC[THREADS];

[kernel_name("mamba3_siso_backward_reverse_p16")]
[domain("ml")]
[variant("short_reverse_p16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	uint p = threadId.x;
	if (bh >= push.b * push.h) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	bool activeP = p < push.p;
	float dState[MAXN];
	if (activeP) {
		for (uint n = 0u; n < push.n; ++n) dState[n] = 0.0f;
	}
	for (uint n = threadId.x; n < push.n; n += THREADS) {
		gsDCBias[n] = 0.0f;
		gsDBBias[n] = 0.0f;
	}
	float dDSum = 0.0f;
	float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint rr = 0u; rr < push.l; ++rr) {
		uint t = push.l - 1u - rr;
		uint tokBH = (b * push.l + t) * push.h + h;
		uint qkBase = tokBH * push.n;
		uint vecBase = tokBH * push.p;
		for (uint n = threadId.x; n < push.n; n += THREADS) {
			gsC[n] = load_f32(storage_buffers[push.crot_idx], qkBase + n);
			gsB[n] = load_f32(storage_buffers[push.brot_idx], qkBase + n);
		}
		GroupMemoryBarrierWithGroupSync();

		float xv = activeP ? load_f32(storage_buffers[push.x_idx], vecBase + p) : 0.0f;
		float dpre = activeP ? load_f32(storage_buffers[push.dpre_idx], vecBase + p) : 0.0f;
		float decay = load_f32(storage_buffers[push.decay_idx], tokBH);
		float scale = load_f32(storage_buffers[push.scale_idx], tokBH);
		float sumB = 0.0f;
		float partA = 0.0f;
		float partScale = 0.0f;
		float dcEven = 0.0f;
		float dcOdd = 0.0f;
		float dbEven = 0.0f;
		float dbOdd = 0.0f;
		for (uint n = 0u; n < push.n; ++n) {
			float dcLocal = 0.0f;
			float dbLocal = 0.0f;
			if (activeP) {
				uint stateIndex = ((bh * push.l + t) * push.p + p) * push.n + n;
				float previous = load_f32(storage_buffers[push.hprev_idx], stateIndex);
				float incoming = dState[n];
				float dhd = dpre * gsC[n] + incoming;
				sumB += incoming * gsB[n];
				partA += dhd * previous;
				partScale += incoming * xv * gsB[n];
				dcLocal = dpre * decay * previous;
				dbLocal = incoming * xv;
				dState[n] = decay * dhd;
			}
			gsReduceA[p] = dcLocal;
			gsReduceB[p] = dbLocal;
			GroupMemoryBarrierWithGroupSync();
			if (p == 0u) {
				float dcSum = 0.0f;
				float dbSum = 0.0f;
				for (uint lane = 0u; lane < THREADS; ++lane) {
					dcSum += gsReduceA[lane];
					dbSum += gsReduceB[lane];
				}
				gsReduceA[0] = dcSum;
				gsReduceB[0] = dbSum;
			}
			GroupMemoryBarrierWithGroupSync();
			float dc = gsReduceA[0];
			float db = gsReduceB[0];
			if (n == 2u * p) {
				dcEven = dc;
				dbEven = db;
			} else if (n == 2u * p + 1u) {
				dcOdd = dc;
				dbOdd = db;
			}
			GroupMemoryBarrierWithGroupSync();
		}
		gsReduceA[p] = activeP ? dpre * xv : 0.0f;
		gsReduceB[p] = partA;
		gsReduceC[p] = partScale;
		GroupMemoryBarrierWithGroupSync();
		if (p == 0u) {
			float dgqSum = 0.0f;
			float daSum = 0.0f;
			float dscaleSum = 0.0f;
			for (uint lane = 0u; lane < THREADS; ++lane) {
				dgqSum += gsReduceA[lane];
				daSum += gsReduceB[lane];
				dscaleSum += gsReduceC[lane];
			}
			gsReduceA[0] = dgqSum;
			gsReduceB[0] = daSum;
			gsReduceC[0] = dscaleSum;
		}
		GroupMemoryBarrierWithGroupSync();
		float totalDGQ = gsReduceA[0];
		float totalDA = gsReduceB[0];
		float totalDScale = gsReduceC[0];

		float dqk = totalDGQ * load_f32(storage_buffers[push.gamma_idx], tokBH);
		if (p < push.a) {
			uint i0 = 2u * p;
			uint i1 = i0 + 1u;
			float dcr0 = dcEven + dqk * gsB[i0];
			float dcr1 = dcOdd + dqk * gsB[i1];
			float dbr0 = scale * dbEven + dqk * gsC[i0];
			float dbr1 = scale * dbOdd + dqk * gsC[i1];
			float th = load_f32(storage_buffers[push.theta_idx], (bh * push.l + t) * push.a + p);
			float cs = cos(th);
			float sn = sin(th);
			float dc0 = dcr0 * cs + dcr1 * sn;
			float dc1 = -dcr0 * sn + dcr1 * cs;
			float db0 = dbr0 * cs + dbr1 * sn;
			float db1 = -dbr0 * sn + dbr1 * cs;
			store_f32(storage_buffers[push.dc_idx], qkBase + i0, dc0);
			store_f32(storage_buffers[push.dc_idx], qkBase + i1, dc1);
			store_f32(storage_buffers[push.db_idx], qkBase + i0, db0);
			store_f32(storage_buffers[push.db_idx], qkBase + i1, db1);
			gsDCBias[i0] += dc0;
			gsDCBias[i1] += dc1;
			gsDBBias[i0] += db0;
			gsDBBias[i1] += db1;
			store_f32(storage_buffers[push.dtheta_idx], (bh * push.l + t) * push.a + p,
				-dcr0 * gsC[i1] + dcr1 * gsC[i0] - dbr0 * gsB[i1] + dbr1 * gsB[i0]);
		}
		uint i0 = 2u * p;
		uint i1 = i0 + 1u;
		if (i0 >= 2u * push.a && i0 < push.n) {
			float dc = dcEven + dqk * gsB[i0];
			float db = scale * dbEven + dqk * gsC[i0];
			store_f32(storage_buffers[push.dc_idx], qkBase + i0, dc);
			store_f32(storage_buffers[push.db_idx], qkBase + i0, db);
			gsDCBias[i0] += dc;
			gsDBBias[i0] += db;
		}
		if (i1 >= 2u * push.a && i1 < push.n) {
			float dc = dcOdd + dqk * gsB[i1];
			float db = scale * dbOdd + dqk * gsC[i1];
			store_f32(storage_buffers[push.dc_idx], qkBase + i1, dc);
			store_f32(storage_buffers[push.db_idx], qkBase + i1, db);
			gsDCBias[i1] += dc;
			gsDBBias[i1] += db;
		}
		if (activeP) {
			store_f32(storage_buffers[push.dx_idx], vecBase + p,
				dpre * (dh + load_f32(storage_buffers[push.gamma_idx], tokBH)
					* load_f32(storage_buffers[push.qk_idx], tokBH)) + scale * sumB);
		}
		if (threadId.x == 0u) {
			store_f32(storage_buffers[push.dadt_idx], tokBH, totalDA * decay);
			store_f32(storage_buffers[push.dscale_idx], tokBH, totalDScale);
			store_f32(storage_buffers[push.dgamma_idx], tokBH,
				totalDGQ * load_f32(storage_buffers[push.qk_idx], tokBH) + totalDScale);
			if (push.has_d != 0u) {
				dDSum += totalDGQ;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (uint n = threadId.x; n < push.n; n += THREADS) {
		store_f32(storage_buffers[push.dcbias_idx], bh * push.n + n, gsDCBias[n]);
		store_f32(storage_buffers[push.dbbias_idx], bh * push.n + n, gsDBBias[n]);
	}
	if (threadId.x == 0u) store_f32(storage_buffers[push.dd_idx], bh, dDSum);
}



