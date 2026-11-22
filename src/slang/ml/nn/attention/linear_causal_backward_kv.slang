// New deterministic suffix adjoint. Lane i owns dK[:,i], dV[:,i].
// Two bounded KV-adjoint vectors replace all saved forward prefixes.
import storage;
import attributes;
import linear_features;
struct PushConstants {
	uint query_index;
	uint key_index;
	uint value_index;
	uint output_index;
	uint denominator_index;
	uint output_gradient_index;
	uint key_gradient_index;
	uint value_gradient_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float epsilon;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("causal_linear_attention_backward_kv")]
[domain("ml")]
[variant("scan_d64_kv")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint head = group_id.x;
	uint i = thread_id.x;
	if (head >= push.batch_heads || i >= push.head_dim) return;
	float g_row[64];
	float g_col[64];
	for (uint j = 0; j < push.head_dim; ++j) { g_row[j] = 0.0f; g_col[j] = 0.0f; }
	float g_z = 0.0f;
	uint base = head * push.sequence_length * push.head_dim;
	for (uint step = 0; step < push.sequence_length; ++step) {
		uint t = push.sequence_length - 1 - step;
		uint row = base + t * push.head_dim;
		float raw_den = load_f32(storage_buffers[push.denominator_index], head * push.sequence_length + t);
		float den = max(raw_den, push.epsilon);
		float q_i = phi(load_f32(storage_buffers[push.query_index], row + i));
		float go_i = load_f32(storage_buffers[push.output_gradient_index], row + i) / den;
		float dot_go_out = 0.0f;
		float dk = 0.0f;
		float dv = 0.0f;
		for (uint j = 0; j < push.head_dim; ++j) {
			float go = load_f32(storage_buffers[push.output_gradient_index], row + j);
			g_row[j] += q_i * go / den;
			g_col[j] += phi(load_f32(storage_buffers[push.query_index], row + j)) * go_i;
			dot_go_out += go * load_f32(storage_buffers[push.output_index], row + j);
			dk += g_row[j] * load_f32(storage_buffers[push.value_index], row + j);
			dv += g_col[j] * phi(load_f32(storage_buffers[push.key_index], row + j));
		}
		if (raw_den >= push.epsilon) g_z -= q_i * dot_go_out / den;
		dk = (dk + g_z) * phi_grad(load_f32(storage_buffers[push.key_index], row + i));
		store_f32(storage_buffers[push.key_gradient_index], row + i, dk);
		store_f32(storage_buffers[push.value_gradient_index], row + i, dv);
	}
}
