// RnnScanBwd — BPTT recurrence scan in ONE dispatch (backward of RnnScan).
//
// One workgroup per batch; loops t = S-1 downto 0, maintaining dH (the gradient
// w.r.t. h^{(t+1)} contributed by later timesteps) in groupshared. Produces
// d_gates_i [B*S,H] and d_gates_h [B*S,H]; the weight/bias gradients are then
// one LinearWeightBiasBwd(hprev, d_gates_h) call (it sums over all B*S rows =
// over t and b), so no separate reduction kernel is needed.
//
// Forward:  h_new = tanh(a),  a = gates_i + gates_h,  gates_h = W_hh @ h_prev + b_hh.
// Given dO_t = dL/dh^{(t+1)} (= upstream output grad + recurrent dH):
//   da = dO * (1 - h_new^2)
//   d_gates_i = da
//   d_gates_h = da
//   dH_prev = da + d_gates_h @ W_hh = da + da @ W_hh

import Activations;
import Storage;

struct PushConstants {
	uint dout_idx;       // [B*S, H] grad w.r.t. out (h^{(t+1)})
	uint gates_i_idx;    // [B*S, H]
	uint hprev_idx;      // [B*S, H] (h^{(t)}, saved by the forward)
	uint w_hh_idx;       // [H, H]
	uint b_hh_idx;       // [H] (present iff has_bias; unused in the math, kept for parity)
	uint d_gates_i_idx;  // [B*S, H] out
	uint d_gates_h_idx;  // [B*S, H] out
	uint hidden_size;    // H
	uint seq_len;        // S
	uint batch;          // B
	uint has_bias;       // 1 if b_hh present
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float dHShared[1024];       // recurrent grad flowing back [H]
groupshared float hNewShared[1024];     // h^{(t+1)} recomputed [H]
groupshared float hpShared[1024];       // h^{(t)} loaded [H]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint S = pc.seq_len;
	if (b >= pc.batch) return;

	// dH = 0 (nothing beyond the last timestep contributes)
	for (uint j0 = lane; j0 < H; j0 += 256) dHShared[j0] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	for (int t = int(S) - 1; t >= 0; --t) {
		uint tu = uint(t);
		uint hpBase = (b * S + tu) * H;
		uint outBase = (b * S + tu) * H;
		uint giBase = (b * S + tu) * H;

		// Load h^{(t)} into shared (needed for gh = W_hh @ h^{(t)} + b_hh recompute).
		for (uint j1a = lane; j1a < H; j1a += 256)
			hpShared[j1a] = OaLoad(heap[pc.hprev_idx], hpBase + j1a);
		GroupMemoryBarrierWithGroupSync();

		// Recompute h^{(t+1)} from saved h^{(t)} and gates_i.
		for (uint j1 = lane; j1 < H; j1 += 256) {
			uint wRow = j1 * H;
			float gh = (pc.has_bias != 0u) ? OaLoad(heap[pc.b_hh_idx], j1) : 0.0f;
			for (uint i = 0; i < H; ++i)
				gh += hpShared[i] * OaLoad(heap[pc.w_hh_idx], wRow + i);
			float gi = OaLoad(heap[pc.gates_i_idx], giBase + j1);
			float hNew = tanh(gh + gi);
			hNewShared[j1] = hNew;
		}
		GroupMemoryBarrierWithGroupSync();

		// da = dO * (1 - h_new^2); d_gates_i = da; d_gates_h = da; dH_prev accumulator = da.
		for (uint j2 = lane; j2 < H; j2 += 256) {
			float dO = OaLoad(heap[pc.dout_idx], outBase + j2) + dHShared[j2];
			float hNew = hNewShared[j2];
			float da = dO * (1.0 - hNew * hNew);
			OaStore(heap[pc.d_gates_i_idx], giBase + j2, da);
			OaStore(heap[pc.d_gates_h_idx], giBase + j2, da);
			hNewShared[j2] = da;  // reuse: accumulator for dH_prev
		}
		GroupMemoryBarrierWithGroupSync();

		// recurrent term: dH_prev[j] = sum_i da[i] * W_hh[i*H+j]
		for (uint j3 = lane; j3 < H; j3 += 256) {
			float sum = 0.0f;
			for (uint i = 0; i < H; ++i)
				sum += hNewShared[i] * OaLoad(heap[pc.w_hh_idx], i * H + j3);
			dHShared[j3] = sum;
		}
		GroupMemoryBarrierWithGroupSync();  // dHShared ready for timestep t-1
	}
}
