// GruScan — whole-sequence GRU recurrent scan in ONE dispatch.
//
// Replaces S serial GruCellLinear dispatches with a single kernel: one workgroup
// per batch element, looping all S timesteps with h^{(t)} held in groupshared.
// Math is identical to GruCellLinear (same gate combine), only the dispatch count
// changes (S -> 1). Used by OaGru when the scan path is enabled.
//
//   for t in 0..S:   (h^{(0)} = 0)
//     gates_h = W_hh @ h^{(t)} + b_hh                  // cooperative over 3H
//     r = sigmoid(gates_i[t][r] + gates_h[r])
//     z = sigmoid(gates_i[t][z] + gates_h[z])
//     n = tanh    (gates_i[t][n] + r * gates_h[n])
//     h^{(t+1)} = (1 - z) * n + z * h^{(t)}
//   out[b,t,:]  = h^{(t+1)}   ([B,S,H])   -- the sequence output
//   hprev[b,t,:]= h^{(t)}     ([B,S,H])   -- saved for the backward weight-grad

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;   // [B*S, 3H], row (b*S + t)
	uint w_hh_idx;      // [3H, H]
	uint b_hh_idx;      // [3H]
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
groupshared float gatesHShared[3072]; // gates_h [3H]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint S = pc.seq_len;
	uint totalGates = 3u * H;
	if (b >= pc.batch) return;

	// h^{(0)} = 0
	for (uint j0 = lane; j0 < H; j0 += 256) hShared[j0] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (uint t = 0; t < S; ++t) {
		// save h^{(t)} to the hprev buffer (hShared currently holds h^{(t)})
		uint hpBase = (b * S + t) * H;
		for (uint j1 = lane; j1 < H; j1 += 256)
			OaStore(heap[pc.hprev_idx], hpBase + j1, hShared[j1]);

		// gates_h = W_hh @ h^{(t)} + b_hh  (cooperative over the 3H gate outputs)
		for (uint idx = lane; idx < totalGates; idx += 256) {
			uint m = idx;
			uint wRow = m * H;
			float sum = (pc.has_bias != 0u) ? OaLoad(heap[pc.b_hh_idx], m) : 0.0f;
			for (uint i = 0; i < H; ++i)
				sum += hShared[i] * OaLoad(heap[pc.w_hh_idx], wRow + i);
			gatesHShared[idx] = sum;
		}
		GroupMemoryBarrierWithGroupSync();

		// pointwise gate combine; write h^{(t+1)} to output and update hShared
		uint giBase = (b * S + t) * totalGates;
		uint outBase = (b * S + t) * H;
		for (uint j2 = lane; j2 < H; j2 += 256) {
			float ir = OaLoad(heap[pc.gates_i_idx], giBase + j2);
			float iz = OaLoad(heap[pc.gates_i_idx], giBase + H + j2);
			float ic = OaLoad(heap[pc.gates_i_idx], giBase + 2u * H + j2);
			float hr = gatesHShared[j2];
			float hz = gatesHShared[H + j2];
			float hn = gatesHShared[2u * H + j2];
			float hp = hShared[j2];
			float r = Sigmoid(ir + hr);
			float z = Sigmoid(iz + hz);
			float n = tanh(ic + r * hn);
			float hNew = (1.0 - z) * n + z * hp;
			hShared[j2] = hNew;
			OaStore(heap[pc.out_idx], outBase + j2, hNew);
		}
		GroupMemoryBarrierWithGroupSync();  // hShared fully updated before next t
	}
}
