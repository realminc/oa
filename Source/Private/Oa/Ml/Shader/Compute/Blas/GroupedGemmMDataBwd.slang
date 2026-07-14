// dX[r,k] = sum_n dY[r,n] * W[e,n,k], tiled over packed rows and K.
import Storage;
struct PushConstants { uint dy_idx; uint w_idx; uint offsets_idx; uint dx_idx; uint R; uint N; uint K; uint E; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
static const uint BM = 32, BO = 32, BR = 16, TM = 4, TN = 4, BR_PAD = BR + 1;
groupshared float smY[BM * BR_PAD];
groupshared float smW[BO * BR_PAD];
uint ExpertForRow(uint r) { uint e = 0; while (e + 1 < pc.E && r >= heap[pc.offsets_idx].Load((e + 1) * 4)) ++e; return e; }
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint rowBase = gid.x * BM, colBase = gid.y * BO;
	if (rowBase >= pc.R || colBase >= pc.K) return;
	uint tid = lid.x, rowEnd = min(rowBase + BM, pc.R);
	uint firstExpert = ExpertForRow(rowBase), lastExpert = ExpertForRow(rowEnd - 1);
	uint threadRow = tid / 8, threadCol = tid % 8;
	uint outRow0 = rowBase + threadRow * TM, outCol0 = colBase + threadCol * TN;
	if (firstExpert != lastExpert) {
		for (uint i = 0; i < TM; ++i) { uint r = outRow0 + i; if (r >= pc.R) continue; uint e = ExpertForRow(r);
			for (uint j = 0; j < TN; ++j) { uint k = outCol0 + j; if (k >= pc.K) continue; float sum = 0.0;
				for (uint n = 0; n < pc.N; ++n) sum += OaLoad(heap[pc.dy_idx], r * pc.N + n) * OaLoad(heap[pc.w_idx], (e * pc.N + n) * pc.K + k);
				OaStore(heap[pc.dx_idx], r * pc.K + k, sum); }} return;
	}
	float acc[TM * TN]; [[unroll]] for (uint i = 0; i < TM * TN; ++i) acc[i] = 0.0;
	for (uint nBase = 0; nBase < pc.N; nBase += BR) {
		[[unroll]] for (uint q = 0; q < 8; ++q) { uint idx = tid + q * 64, rr = idx / BR, nn = idx % BR; uint gr = rowBase + rr, gn = nBase + nn;
			smY[rr * BR_PAD + nn] = (gr < pc.R && gn < pc.N) ? OaLoad(heap[pc.dy_idx], gr * pc.N + gn) : 0.0;
			uint gk = colBase + rr; smW[rr * BR_PAD + nn] = (gk < pc.K && gn < pc.N) ? OaLoad(heap[pc.w_idx], (firstExpert * pc.N + gn) * pc.K + gk) : 0.0; }
		GroupMemoryBarrierWithGroupSync();
		[[unroll]] for (uint nn = 0; nn < BR; ++nn) { float yv[TM], wv[TN];
			[[unroll]] for (uint i = 0; i < TM; ++i) yv[i] = smY[(threadRow * TM + i) * BR_PAD + nn];
			[[unroll]] for (uint j = 0; j < TN; ++j) wv[j] = smW[(threadCol * TN + j) * BR_PAD + nn];
			[[unroll]] for (uint i = 0; i < TM; ++i) [[unroll]] for (uint j = 0; j < TN; ++j) acc[i * TN + j] += yv[i] * wv[j]; }
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) { uint r = outRow0 + i, k = outCol0 + j; if (r < pc.R && k < pc.K) OaStore(heap[pc.dx_idx], r * pc.K + k, acc[i * TN + j]); }
}
