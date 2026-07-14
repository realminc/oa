// GroupedGemmM shares the tiled GroupedLinearM layout without a bias epilogue.
import Storage;
struct PushConstants { uint x_idx; uint w_idx; uint offsets_idx; uint y_idx; uint R; uint N; uint K; uint E; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
static const uint BM = 32, BN = 32, BK = 16, TM = 4, TN = 4, BK_PAD = BK + 1;
groupshared float smX[BM * BK_PAD];
groupshared float smW[BN * BK_PAD];
uint ExpertForRow(uint r) { uint e = 0; while (e + 1 < pc.E && r >= heap[pc.offsets_idx].Load((e + 1) * 4)) ++e; return e; }
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint rowBase = gid.x * BM, colBase = gid.y * BN;
	if (rowBase >= pc.R || colBase >= pc.N) return;
	uint tid = lid.x, rowEnd = min(rowBase + BM, pc.R);
	uint firstExpert = ExpertForRow(rowBase), lastExpert = ExpertForRow(rowEnd - 1);
	uint threadRow = tid / 8, threadCol = tid % 8;
	uint outRow0 = rowBase + threadRow * TM, outCol0 = colBase + threadCol * TN;
	if (firstExpert != lastExpert) {
		for (uint i = 0; i < TM; ++i) { uint r = outRow0 + i; if (r >= pc.R) continue; uint e = ExpertForRow(r);
			for (uint j = 0; j < TN; ++j) { uint n = outCol0 + j; if (n >= pc.N) continue; float sum = 0.0;
				for (uint k = 0; k < pc.K; ++k) sum += OaLoad(heap[pc.x_idx], r * pc.K + k) * OaLoad(heap[pc.w_idx], (e * pc.N + n) * pc.K + k);
				OaStore(heap[pc.y_idx], r * pc.N + n, sum); }} return;
	}
	float acc[TM * TN]; [[unroll]] for (uint i = 0; i < TM * TN; ++i) acc[i] = 0.0;
	for (uint kBase = 0; kBase < pc.K; kBase += BK) {
		[[unroll]] for (uint q = 0; q < 8; ++q) { uint idx = tid + q * 64, rr = idx / BK, kk = idx % BK; uint gr = rowBase + rr, gk = kBase + kk;
			smX[rr * BK_PAD + kk] = (gr < pc.R && gk < pc.K) ? OaLoad(heap[pc.x_idx], gr * pc.K + gk) : 0.0;
			uint gn = colBase + rr; smW[rr * BK_PAD + kk] = (gn < pc.N && gk < pc.K) ? OaLoad(heap[pc.w_idx], (firstExpert * pc.N + gn) * pc.K + gk) : 0.0; }
		GroupMemoryBarrierWithGroupSync();
		[[unroll]] for (uint kk = 0; kk < BK; ++kk) { float xv[TM], wv[TN];
			[[unroll]] for (uint i = 0; i < TM; ++i) xv[i] = smX[(threadRow * TM + i) * BK_PAD + kk];
			[[unroll]] for (uint j = 0; j < TN; ++j) wv[j] = smW[(threadCol * TN + j) * BK_PAD + kk];
			[[unroll]] for (uint i = 0; i < TM; ++i) [[unroll]] for (uint j = 0; j < TN; ++j) acc[i * TN + j] += xv[i] * wv[j]; }
		GroupMemoryBarrierWithGroupSync();
	}
	for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) { uint r = outRow0 + i, n = outCol0 + j; if (r < pc.R && n < pc.N) OaStore(heap[pc.y_idx], r * pc.N + n, acc[i * TN + j]); }
}
