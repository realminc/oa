// GruCellLinear — fused GRU recurrent cell: hidden projection + pointwise gate combine.
//
// Replaces the per-timestep pair of dispatches (Linear(h, W_hh) + GruCellPointwise)
// with one kernel over [B, H]. Each workgroup handles one batch element; threads
// compute the [3H] hidden-projection vector (gates_h) and then apply the pointwise
// GRU update. When save_gates_h is set, the hidden projection is also written out
// so the existing unchanged GruCellPointwiseBwd + LinearBwd backward path can be
// reused.
//
//   gates_h = h_prev @ W_hh^T + b_hh
//   r     = sigmoid(gates_i[r] + gates_h[r])
//   z     = sigmoid(gates_i[z] + gates_h[z])
//   n     = tanh    (gates_i[n] + r * gates_h[n])
//   h_new = (1 - z) * n + z * h_prev

import Activations;
import Storage;

struct PushConstants {
	uint h_idx;        // h_prev [B, H]
	uint w_hh_idx;     // W_hh [3H, H]
	uint b_hh_idx;     // b_hh [3H]
	uint gates_i_idx;  // gatesI [B*T, 3H]
	uint out_idx;      // h_new [B, H]
	uint gates_h_idx;  // gates_h [B, 3H] (written when save_gates_h != 0)
	uint hidden_size;  // H
	uint batch;        // B
	uint time_offset;  // timestep offset into gates_i (in rows of 3H)
	uint batch_stride; // row stride between batches in gates_i (T for batch-major, 1 for contiguous)
	uint has_bias;     // 1 if b_hh present
	uint save_gates_h; // 1 to write gates_h out for the unchanged backward path
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Max supported hidden size: 1024. groupshared size is 3 * 1024 * 4 = 12 KiB.
groupshared float gatesHShared[3072];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint b = gid.x;
	uint lane = ltid.x;
	uint H = pc.hidden_size;
	uint B = pc.batch;
	uint totalGates = 3u * H;
	if (b >= B) return;

	uint hBase = b * H;

	// Step 1: each thread computes a subset of gates_h entries for this batch.
	// gates_h[k*H + j] = sum_i h_prev[i] * W_hh[(k*H + j)*H + i] + b_hh[k*H + j]
	for (uint idx = lane; idx < totalGates; idx += 256) {
		uint k = idx / H;
		uint j = idx - k * H;
		uint wRow = (k * H + j) * H;
		float sum = 0.0f;
		for (uint i = 0; i < H; ++i) {
			float hVal = OaLoad(heap[pc.h_idx], hBase + i);
			sum += hVal * OaLoad(heap[pc.w_hh_idx], wRow + i);
		}
		if (pc.has_bias != 0u) sum += OaLoad(heap[pc.b_hh_idx], idx);
		gatesHShared[idx] = sum;
	}

	GroupMemoryBarrierWithGroupSync();

	// Step 2: each thread computes one or more h_new elements for this batch.
	uint giBase = (b * pc.batch_stride + pc.time_offset) * totalGates;
	uint ghBase = b * totalGates;
	for (uint j = lane; j < H; j += 256) {
		float ir = OaLoad(heap[pc.gates_i_idx], giBase + j);
		float iz = OaLoad(heap[pc.gates_i_idx], giBase + j + H);
		float ic = OaLoad(heap[pc.gates_i_idx], giBase + j + 2u * H);
		float hr = gatesHShared[j];
		float hz = gatesHShared[j + H];
		float hn = gatesHShared[j + 2u * H];
		float hp = OaLoad(heap[pc.h_idx], hBase + j);

		float r = Sigmoid(ir + hr);
		float z = Sigmoid(iz + hz);
		float n = tanh(ic + r * hn);
		float hNew = (1.0 - z) * n + z * hp;

		OaStore(heap[pc.out_idx], hBase + j, hNew);

		if (pc.save_gates_h != 0u) {
			OaStore(heap[pc.gates_h_idx], ghBase + j, hr);
			OaStore(heap[pc.gates_h_idx], ghBase + j + H, hz);
			OaStore(heap[pc.gates_h_idx], ghBase + j + 2u * H, hn);
		}
	}
}
