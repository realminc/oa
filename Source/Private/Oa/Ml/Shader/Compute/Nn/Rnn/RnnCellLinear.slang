// RnnCellLinear — fused vanilla RNN recurrent cell: hidden projection + tanh combine.
//
// Replaces the per-timestep pair of dispatches (Linear(h, W_hh) + RnnCellPointwise)
// with one kernel over [B, H]. Each workgroup handles one batch element; threads
// compute the [H] hidden-projection vector (gh) and then apply tanh(gi + gh).
//
//   gh = h_prev @ W_hh^T + b_hh
//   h_new = tanh(gi + gh)

import Activations;
import Storage;

struct PushConstants {
	uint h_idx;        // h_prev [B, H]
	uint w_hh_idx;     // W_hh [H, H]
	uint b_hh_idx;     // b_hh [H]
	uint gi_idx;       // gi [B*T, H] — input projection for the whole sequence
	uint out_idx;      // h_new [B, H]
	uint gh_idx;       // gh [B, H] (written when save_gh != 0)
	uint hidden_size;  // H
	uint batch;        // B
	uint time_offset;  // timestep offset into gi (in rows of H)
	uint batch_stride; // row stride between batches in gi (T for batch-major, 1 for contiguous)
	uint has_bias;     // 1 if b_hh present
	uint save_gh;      // 1 to write gh out for the unchanged backward path
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Max supported hidden size: 1024. groupshared size is 1024 * 4 = 4 KiB.
groupshared float ghShared[1024];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint B = pc.batch;
	if (b >= B) return;

	uint hBase = b * H;

	// Step 1: each thread computes a subset of gh entries for this batch.
	// gh[j] = sum_i h_prev[i] * W_hh[j*H + i] + b_hh[j]
	for (uint j = lane; j < H; j += 256) {
		uint wRow = j * H;
		float sum = 0.0f;
		for (uint i = 0; i < H; ++i) {
			float hVal = OaLoad(heap[pc.h_idx], hBase + i);
			sum += hVal * OaLoad(heap[pc.w_hh_idx], wRow + i);
		}
		if (pc.has_bias != 0u) sum += OaLoad(heap[pc.b_hh_idx], j);
		ghShared[j] = sum;
	}

	GroupMemoryBarrierWithGroupSync();

	// Step 2: each thread computes one or more h_new elements for this batch.
	// gi is the whole-sequence projection [B*T, H]; index the row for this (b, t)
	// directly via time_offset/batch_stride instead of slicing it per timestep.
	uint giBase = (b * pc.batch_stride + pc.time_offset) * H;
	for (uint j = lane; j < H; j += 256) {
		float gi = OaLoad(heap[pc.gi_idx], giBase + j);
		float gh = ghShared[j];
		float hNew = tanh(gi + gh);
		OaStore(heap[pc.out_idx], hBase + j, hNew);

		if (pc.save_gh != 0u) {
			OaStore(heap[pc.gh_idx], hBase + j, gh);
		}
	}
}
