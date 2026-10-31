// New provider: no OA C++ linear-attention donor exists (see oaAttention.md).
// Each head owns a workgroup; lane j owns KV[:,j] and output[:,j].
// Private bounded state is O(d^2), retained global state is only O(S*d).
import storage;
import attributes;
import linear_features;

struct PushConstants {
	uint query_index;
	uint key_index;
	uint value_index;
	uint output_index;
	uint denominator_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float epsilon;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];


[kernel_name("causal_linear_attention")]
[domain("ml")]
[variant("scan_d64")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint head = group_id.x;
	uint col = thread_id.x;
	if (head >= push.batch_heads || col >= push.head_dim) return;
	float kv[64];
	float z[64];
	for (uint i = 0; i < push.head_dim; ++i) { kv[i] = 0.0f; z[i] = 0.0f; }
	uint base = head * push.sequence_length * push.head_dim;
	for (uint t = 0; t < push.sequence_length; ++t) {
		uint row = base + t * push.head_dim;
		float v = load_f32(storage_buffers[push.value_index], row + col);
		float numerator = 0.0f;
		float denominator = 0.0f;
		for (uint i = 0; i < push.head_dim; ++i) {
			float k = phi(load_f32(storage_buffers[push.key_index], row + i));
			float q = phi(load_f32(storage_buffers[push.query_index], row + i));
			kv[i] += k * v;
			z[i] += k;
			numerator += q * kv[i];
			denominator += q * z[i];
		}
		store_f32(storage_buffers[push.output_index], row + col, numerator / max(denominator, push.epsilon));
		if (col == 0) store_f32(storage_buffers[push.denominator_index], head * push.sequence_length + t, denominator);
	}
}
