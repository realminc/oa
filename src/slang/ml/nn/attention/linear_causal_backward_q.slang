// New deterministic adjoint. Lane i recomputes its prefix KV row, owns dQ[:,i].
// No prefix checkpoints or floating-point atomics.
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
	uint query_gradient_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float epsilon;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("causal_linear_attention_backward")]
[domain("ml")]
[variant("scan_d64_q")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint head = group_id.x;
	uint i = thread_id.x;
	if (head >= push.batch_heads || i >= push.head_dim) return;
	float kv[64];
	for (uint j = 0; j < push.head_dim; ++j) kv[j] = 0.0f;
	float z = 0.0f;
	uint base = head * push.sequence_length * push.head_dim;
	for (uint t = 0; t < push.sequence_length; ++t) {
		uint row = base + t * push.head_dim;
		float k = phi(load_f32(storage_buffers[push.key_index], row + i));
		z += k;
		float raw_den = load_f32(storage_buffers[push.denominator_index], head * push.sequence_length + t);
		float den = max(raw_den, push.epsilon);
		float dq = 0.0f;
		float dot_go_out = 0.0f;
		for (uint j = 0; j < push.head_dim; ++j) {
			kv[j] += k * load_f32(storage_buffers[push.value_index], row + j);
			float go = load_f32(storage_buffers[push.output_gradient_index], row + j);
			dq += go * kv[j];
			dot_go_out += go * load_f32(storage_buffers[push.output_index], row + j);
		}
		if (raw_den >= push.epsilon) dq -= dot_go_out * z;
		dq *= phi_grad(load_f32(storage_buffers[push.query_index], row + i)) / den;
		store_f32(storage_buffers[push.query_gradient_index], row + i, dq);
	}
}
