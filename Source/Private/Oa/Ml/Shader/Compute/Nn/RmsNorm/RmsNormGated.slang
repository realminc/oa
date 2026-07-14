import Reductions;
import Storage;
import Activations;  // Silu — shared module (was inline z*sigmoid(z); byte-identical)

struct PushConstants {
	uint x_idx;
	uint weight_idx;
	uint bias_idx;
	uint z_idx;
	uint out_idx;
	uint rows;
	uint cols;
	float eps;
	uint has_bias;
	uint has_z;
	uint norm_before_gate;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;

	uint lane = tid.x;
	uint base = row * pc.cols;

	// Load x
	float inv_rms = InvRms(heap[pc.x_idx], base, pc.cols, pc.eps, lane);

	// Load z if present and apply gate before norm
	if (pc.has_z != 0 && pc.norm_before_gate == 0) {
		for (uint i = lane; i < pc.cols; i += 256) {
			float z_val = OaLoad(heap[pc.z_idx], base + i);
			float x_val = OaLoad(heap[pc.x_idx], base + i);
			float gated = x_val * Silu(z_val); // SiLU
			OaStore(heap[pc.x_idx], base + i, gated);
		}
		// Recompute inv_rms on gated values
		inv_rms = InvRms(heap[pc.x_idx], base, pc.cols, pc.eps, lane);
	}

	// Normalize and scale
	for (uint i = lane; i < pc.cols; i += 256) {
		float x_val = OaLoad(heap[pc.x_idx], base + i);
		float w_val = OaLoad(heap[pc.weight_idx], i);
		float normed = x_val * inv_rms * w_val;

		// Add bias if present
		if (pc.has_bias != 0) {
			float b_val = OaLoad(heap[pc.bias_idx], i);
			normed += b_val;
		}

		// Apply gate after norm if norm_before_gate
		if (pc.has_z != 0 && pc.norm_before_gate != 0) {
			float z_val = OaLoad(heap[pc.z_idx], base + i);
			normed *= Silu(z_val); // SiLU
		}

		OaStore(heap[pc.out_idx], base + i, normed);
	}
}
