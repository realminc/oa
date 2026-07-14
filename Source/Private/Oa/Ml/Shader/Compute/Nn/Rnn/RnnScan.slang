// RnnScan — whole-sequence vanilla RNN recurrent scan in ONE dispatch.
//
// Replaces S serial RnnCellLinear dispatches with a single kernel: one workgroup
// per batch element, looping all S timesteps with h^{(t)} held in groupshared.
// Math is identical to RnnCellLinear (same tanh combine), only the dispatch count
// changes (S -> 1). Used by OaRnn when the scan path is enabled.
//
//   for t in 0..S:   (h^{(0)} = 0)
//     a = W_hh @ h^{(t)} + b_hh + gi[t]
//     h^{(t+1)} = tanh(a)
//   out[b,t,:]  = h^{(t+1)}   ([B,S,H])   -- the sequence output
//   hprev[b,t,:]= h^{(t)}     ([B,S,H])   -- saved for the backward weight-grad

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;   // [B*S, H], row (b*S + t)
	uint w_hh_idx;      // [H, H]
	uint b_hh_idx;      // [H]
	uint out_idx;       // [B*S, H] (== [B,S,H] layout)
	uint hprev_idx;     // [B*S, H] (h^{(t)}, saved)
	uint hidden_size;   // H
	uint seq_len;       // S
	uint batch;         // B
	uint has_bias;      // 1 if b_hh present
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float hShared[1024];      // h^{(t)} [H]
groupshared float aShared[1024];      // pre-tanh a [H]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint S = pc.seq_len;
	if (b >= pc.batch) return;

	// h^{(0)} = 0
	for (uint j0 = lane; j0 < H; j0 += 256) hShared[j0] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint t = 0; t < S; ++t) {
		// save h^{(t)} to the hprev buffer (hShared currently holds h^{(t)})
		uint hpBase = (b * S + t) * H;
		for (uint j1 = lane; j1 < H; j1 += 256)
			OaStore(heap[pc.hprev_idx], hpBase + j1, hShared[j1]);

		// a = W_hh @ h^{(t)} + b_hh + gi[t]  (cooperative over H)
		for (uint j2 = lane; j2 < H; j2 += 256) {
			uint wRow = j2 * H;
			float sum = (pc.has_bias != 0u) ? OaLoad(heap[pc.b_hh_idx], j2) : 0.0f;
			for (uint i = 0; i < H; ++i)
				sum += hShared[i] * OaLoad(heap[pc.w_hh_idx], wRow + i);
			aShared[j2] = sum;
		}
		GroupMemoryBarrierWithGroupSync();

		uint giBase = (b * S + t) * H;
		uint outBase = (b * S + t) * H;
		for (uint j3 = lane; j3 < H; j3 += 256) {
			float gi = OaLoad(heap[pc.gates_i_idx], giBase + j3);
			float a = aShared[j3] + gi;
			float hNew = tanh(a);
			hShared[j3] = hNew;
			OaStore(heap[pc.out_idx], outBase + j3, hNew);
		}
		GroupMemoryBarrierWithGroupSync();  // hShared fully updated before next t
	}
}
