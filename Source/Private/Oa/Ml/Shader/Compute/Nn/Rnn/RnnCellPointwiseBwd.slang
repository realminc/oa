// RnnCellPointwiseBwd — fused vanilla-RNN pointwise backward.
//
// Forward:  h_new = tanh(a),  a = gates_i + gates_h
// Backward (chain rule): with dOut = dL/dh_new and h = tanh(a),
//   da  = dOut * (1 - h^2)
//   d_gates_i = d_gates_h = da
//
// h is recomputed from the saved gate projections (cheap, avoids saving it).

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;
	uint gates_h_idx;
	uint dout_idx;        // dL/dh_new [B, H]
	uint d_gates_i_idx;   // [B, H] (or [B*T, H] with time_offset)
	uint d_gates_h_idx;   // [B, H]
	uint hidden_size;     // H
	uint count;           // B * H
	uint time_offset;     // timestep offset into gates_i / d_gates_i (in rows of H)
	uint batch_stride;    // row stride between batches in gates_i (T for batch-major, 1 for contiguous)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	uint H = pc.hidden_size;
	uint b = idx / H;
	uint j = idx - b * H;
	uint gi = (b * pc.batch_stride + pc.time_offset) * H + j; // gates_i: batch-major [B*T, H]
	uint gh = b * H + j;                                      // gates_h / dout / d_gates_h: [B, H]

	float giVal = OaLoad(heap[pc.gates_i_idx], gi);
	float ghVal = OaLoad(heap[pc.gates_h_idx], gh);
	float h = tanh(giVal + ghVal);

	float dO = OaLoad(heap[pc.dout_idx], gh);
	float da = dO * (1.0 - h * h);

	OaStore(heap[pc.d_gates_i_idx], gi, da);
	OaStore(heap[pc.d_gates_h_idx], gh, da);
}
