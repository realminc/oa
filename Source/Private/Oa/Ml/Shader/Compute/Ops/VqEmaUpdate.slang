// VqEmaUpdate — EMA codebook update + dead-code reinit for VQ-VAE (van den Oord 2017).
//
// The codebook is NOT trained by gradient. Instead each code tracks the running
// mean of the encoder outputs assigned to it:
//   N_k ← γ·N_k + (1-γ)·count_k                     (EMA cluster size)
//   m_k ← γ·m_k + (1-γ)·Σ_{n: idx[n]=k} z_e[n]      (EMA cluster sum)
//   e_k  = m_k / max(N_k, eps)                        (codebook entry)
// Dead codes (N_k < dead_threshold) are revived from a live encoder row so the
// codebook can't collapse to a handful of entries.
//
// One thread per code k. Each scans all N rows (O(K·D·N), trivial at these sizes).
// No atomics. Buffers: ze[N,D] R, idx[N] R, embed_sum[K,D] RW, cluster_size[K] RW,
// codebook[K,D] W.

import Storage;

struct PushConstants {
	uint ze_idx;
	uint idx_idx;
	uint embed_sum_idx;
	uint cluster_size_idx;
	uint codebook_idx;
	uint N;             // rows (latent vectors this batch)
	uint D;             // latent dim
	uint K;             // codebook size
	float decay;        // EMA γ (e.g. 0.99)
	float eps;          // division floor
	float dead_thresh;  // revive codes with N_k below this
	uint  seed;         // per-step seed (mixes into revival row → scatter, not re-die)
	uint  normalize;    // 1 = rescale each codebook row to unit RMS (cosine VQ, anti-collapse)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint k = tid.x;
	if (k >= pc.K) return;

	// Count rows assigned to code k this batch.
	uint cnt = 0u;
	for (uint n = 0u; n < pc.N; ++n) {
		int a = int(heap[pc.idx_idx].Load(n * 4u));
		if (uint(a) == k) ++cnt;
	}

	// EMA cluster size.
	float Nk = pc.decay * OaLoadF32(heap[pc.cluster_size_idx], k) + (1.0F - pc.decay) * float(cnt);

	// EMA cluster sum + codebook write, one latent dim at a time (keeps registers low).
	// Accumulate the sum-of-squares of the codebook row so it can be rescaled to unit
	// RMS afterwards (cosine VQ): the EMA mean of unit-RMS latents shrinks toward the
	// centroid, which collapses assignment onto one near-centroid code — renormalizing
	// keeps every code on the same sphere as z_e so direction (not magnitude) decides.
	uint cbBase = k * pc.D;
	float sumsq = 0.0F;
	for (uint d = 0u; d < pc.D; ++d) {
		float s = 0.0F;
		for (uint n = 0u; n < pc.N; ++n) {
			int a = int(heap[pc.idx_idx].Load(n * 4u));
			if (uint(a) == k) { s += OaLoad(heap[pc.ze_idx], n * pc.D + d); }
		}
		float mk = pc.decay * OaLoad(heap[pc.embed_sum_idx], cbBase + d)
		         + (1.0F - pc.decay) * s;
		OaStore(heap[pc.embed_sum_idx], cbBase + d, mk);   // EMA state stays in raw z_e units
		float ek = mk / max(Nk, pc.eps);
		OaStore(heap[pc.codebook_idx], cbBase + d, ek);
		sumsq += ek * ek;
	}
	OaStoreF32(heap[pc.cluster_size_idx], k, Nk);

	// Rescale the codebook row (readout only — embed_sum keeps the raw EMA mean) to unit
	// RMS so ||e_k|| matches the encoder's unit-RMS z_e and L2 assignment becomes cosine.
	if (pc.normalize != 0u) {
		float rms = sqrt(sumsq / float(pc.D));
		if (rms > pc.eps) {
			float inv = 1.0F / rms;
			for (uint d = 0u; d < pc.D; ++d) {
				OaStore(heap[pc.codebook_idx], cbBase + d, OaLoad(heap[pc.codebook_idx], cbBase + d) * inv);
			}
		}
	}

	// Dead-code reinit: revive from a STOCHASTIC live encoder row. The row index mixes
	// the code id with a per-step seed, so a code that's revived onto already-claimed
	// territory (and loses again) lands on a DIFFERENT row next step instead of re-dying
	// on the same deterministic row forever. Over steps this scatters dead codes across
	// the encoder's output distribution until they catch genuinely uncovered regions.
	if (Nk < pc.dead_thresh) {
		uint h = ((k + 1u) * 2654435761u) ^ (pc.seed * 2246822519u) ^ (pc.seed >> 15);
		h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;   // finalize for good low-bit mixing
		uint r = h % pc.N;
		uint rBase = r * pc.D;
		for (uint d = 0u; d < pc.D; ++d) {
			float v = OaLoad(heap[pc.ze_idx], rBase + d);
			OaStore(heap[pc.codebook_idx], cbBase + d, v);
			OaStore(heap[pc.embed_sum_idx], cbBase + d, v);
		}
		OaStoreF32(heap[pc.cluster_size_idx], k, 1.0F);
	}
}
