// Complete independent-chunk Mamba-3 reverse pass.
//
// Forward H[t-1] history and reverse exit adjoints make the 16-token chunks
// independent. Each value-row lane keeps its state adjoint private while
// workgroup reductions combine value rows for C/B and scalar derivatives.

import storage;
import attributes;

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
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint a;
	uint chunks;
	uint has_d;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

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

[kernel_name("mamba3_siso_chunk_backward")]
[domain("ml")]
[variant("chunk16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / push.chunks;
	uint chunk = group - bh * push.chunks;
	uint p = threadId.x;
	if (bh >= push.b * push.h) return;
	bool active = p < push.p;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, push.l);
	uint chunkState = ((bh * push.chunks + chunk) * push.p + p) * push.n;
	float dState[MAXN];
	if (active) {
		for (uint n = 0u; n < push.n; ++n) {
			dState[n] = load_f32(storage_buffers[push.exit_idx], chunkState + n);
		}
	}
	float direct = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;

	for (uint rr = 0u; rr < end - begin; ++rr) {
		uint t = end - 1u - rr;
		uint scalar = (b * push.l + t) * push.h + h;
		uint vector = scalar * push.p + p;
		uint stateBase = scalar * push.n;
		float hPrev[MAXN];
		if (active) {
			for (uint n = 0u; n < push.n; ++n) {
				uint history = ((bh * push.l + t) * push.p + p) * push.n + n;
				hPrev[n] = load_f32(storage_buffers[push.hprev_idx], history);
			}
		}

		for (uint n = threadId.x; n < push.n; n += THREADS) {
			gsC[n] = load_f32(storage_buffers[push.crot_idx], stateBase + n);
			gsB[n] = load_f32(storage_buffers[push.brot_idx], stateBase + n);
		}
		GroupMemoryBarrierWithGroupSync();
		float x = active ? load_f32(storage_buffers[push.x_idx], vector) : 0.0f;
		float dpre = active ? load_f32(storage_buffers[push.dpre_idx], vector) : 0.0f;
		float decay = load_f32(storage_buffers[push.decay_idx], scalar);
		float localDAdt = 0.0f;
		float localDScale = 0.0f;
		float sumB = 0.0f;
		for (uint n = 0u; n < push.n; ++n) {
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

		if (threadId.x < push.n) {
			float dc = 0.0f;
			float db = 0.0f;
			for (uint row = 0u; row < push.p; ++row) {
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
			for (uint row = 0u; row < push.p; ++row) {
				gsDGQ += gsDGQPart[row];
				gsDAdt += gsDAdtPart[row];
				gsDScale += gsDScalePart[row];
			}
		}
		GroupMemoryBarrierWithGroupSync();

		float dqk = gsDGQ * load_f32(storage_buffers[push.gamma_idx], scalar);
		for (uint k = threadId.x; k < push.a; k += THREADS) {
			uint i0 = 2u * k;
			uint i1 = i0 + 1u;
			float dcr0 = gsDCSum[i0] + dqk * gsB[i0];
			float dcr1 = gsDCSum[i1] + dqk * gsB[i1];
			float dbr0 = load_f32(storage_buffers[push.scale_idx], scalar) * gsDBSum[i0]
				+ dqk * gsC[i0];
			float dbr1 = load_f32(storage_buffers[push.scale_idx], scalar) * gsDBSum[i1]
				+ dqk * gsC[i1];
			float theta = load_f32(storage_buffers[push.theta_idx], (bh * push.l + t) * push.a + k);
			float cs = cos(theta);
			float sn = sin(theta);
			float dc0 = dcr0 * cs + dcr1 * sn;
			float dc1 = -dcr0 * sn + dcr1 * cs;
			float db0 = dbr0 * cs + dbr1 * sn;
			float db1 = -dbr0 * sn + dbr1 * cs;
			store_f32(storage_buffers[push.dc_idx], stateBase + i0, dc0);
			store_f32(storage_buffers[push.dc_idx], stateBase + i1, dc1);
			store_f32(storage_buffers[push.db_idx], stateBase + i0, db0);
			store_f32(storage_buffers[push.db_idx], stateBase + i1, db1);
			store_f32(storage_buffers[push.dtheta_idx], (bh * push.l + t) * push.a + k,
				-dcr0 * gsC[i1] + dcr1 * gsC[i0]
				-dbr0 * gsB[i1] + dbr1 * gsB[i0]);
		}
		for (uint n = 2u * push.a + threadId.x; n < push.n; n += THREADS) {
			float dc = gsDCSum[n] + dqk * gsB[n];
			float db = load_f32(storage_buffers[push.scale_idx], scalar) * gsDBSum[n]
				+ dqk * gsC[n];
			store_f32(storage_buffers[push.dc_idx], stateBase + n, dc);
			store_f32(storage_buffers[push.db_idx], stateBase + n, db);
		}
		if (active) {
			store_f32(storage_buffers[push.dx_idx], vector,
				dpre * (direct + load_f32(storage_buffers[push.gamma_idx], scalar)
					* load_f32(storage_buffers[push.qk_idx], scalar))
				+ load_f32(storage_buffers[push.scale_idx], scalar) * sumB);
			for (uint n = 0u; n < push.n; ++n) {
				dState[n] = decay * (dpre * gsC[n] + dState[n]);
			}
		}
		if (threadId.x == 0u) {
			store_f32(storage_buffers[push.dadt_idx], scalar, gsDAdt * decay);
			store_f32(storage_buffers[push.dgamma_idx], scalar,
				gsDGQ * load_f32(storage_buffers[push.qk_idx], scalar) + gsDScale);
			store_f32(storage_buffers[push.dscale_idx], scalar, gsDScale);
			store_f32(storage_buffers[push.ddtoken_idx], scalar,
				push.has_d != 0u ? gsDGQ : 0.0f);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
