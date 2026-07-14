// GruCellPointwise — fused GRU pointwise gate combine (forward).
//
// Replaces ~10 elementwise dispatches (Slice×6 + Sigmoid×2 + Tanh + Mul×2 +
// Add/Sub×3) with a single per-element kernel over [B, H].
//
//   r     = sigmoid(gates_i[r] + gates_h[r])
//   z     = sigmoid(gates_i[z] + gates_h[z])
//   n     = tanh    (gates_i[n] + r * gates_h[n])
//   h_new = (1 - z) * n + z * h_prev
//
// gates_i / gates_h are [B, 3H] row-major with reset|update|candidate along dim 1:
//   [., 0:H]=reset(r), [., H:2H]=update(z), [., 2H:3H]=candidate(n).

import Activations;
import Storage;

struct PushConstants {
	uint gates_i_idx;
	uint gates_h_idx;
	uint hidden_idx;   // h_prev [B, H]
	uint out_idx;      // h_new [B, H]
	uint hidden_size;  // H
	uint count;        // B * H threads
	uint time_offset;  // timestep offset into gates_i (in rows of 3H)
	uint batch_stride; // row stride between batches in gates_i (T for batch-major, 1 for contiguous)
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
	uint j = idx - b * H;       // idx % H
	uint gi = (b * pc.batch_stride + pc.time_offset) * (3u * H) + j; // gates_i: batch-major [B*T, 3H]
	uint gh3 = b * (3u * H) + j;    // gates_h: [B, 3H] (no time dimension)
	uint gh = b * H + j;            // hidden / out: [B, H]

	float ir = OaLoad(heap[pc.gates_i_idx], gi);
	float iz = OaLoad(heap[pc.gates_i_idx], gi + H);
	float ic = OaLoad(heap[pc.gates_i_idx], gi + 2u * H);
	float hr = OaLoad(heap[pc.gates_h_idx], gh3);
	float hz = OaLoad(heap[pc.gates_h_idx], gh3 + H);
	float hn = OaLoad(heap[pc.gates_h_idx], gh3 + 2u * H);
	float hp = OaLoad(heap[pc.hidden_idx], gh);

	float r = Sigmoid(ir + hr);
	float z = Sigmoid(iz + hz);
	float n = tanh(ic + r * hn);
	float h_new = (1.0 - z) * n + z * hp;

	OaStore(heap[pc.out_idx], gh, h_new);
}
