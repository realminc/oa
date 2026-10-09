// Pointwise portion of one GRU cell adjoint.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint gates_i_index;
	uint gates_h_index;
	uint hidden_index;
	uint output_gradient_index;
	uint gates_i_gradient_index;
	uint gates_h_gradient_index;
	uint direct_hidden_gradient_index;
	uint hidden_size;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gru_cell_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	uint batch_index = index / push.hidden_size;
	uint feature = index % push.hidden_size;
	uint gate_base = batch_index * (3u * push.hidden_size);

	float input_reset = load_f32(storage_buffers[push.gates_i_index], gate_base + feature);
	float input_update = load_f32(
		storage_buffers[push.gates_i_index], gate_base + push.hidden_size + feature
	);
	float input_candidate = load_f32(
		storage_buffers[push.gates_i_index], gate_base + 2u * push.hidden_size + feature
	);
	float hidden_reset = load_f32(storage_buffers[push.gates_h_index], gate_base + feature);
	float hidden_update = load_f32(
		storage_buffers[push.gates_h_index], gate_base + push.hidden_size + feature
	);
	float hidden_candidate = load_f32(
		storage_buffers[push.gates_h_index], gate_base + 2u * push.hidden_size + feature
	);
	float previous = load_f32(storage_buffers[push.hidden_index], index);
	float reset = sigmoid(input_reset + hidden_reset);
	float update = sigmoid(input_update + hidden_update);
	float candidate = tanh(input_candidate + reset * hidden_candidate);
	float upstream = load_f32(storage_buffers[push.output_gradient_index], index);
	float candidate_logit_gradient = upstream * (1.0f - update)
		* (1.0f - candidate * candidate);
	float reset_logit_gradient = candidate_logit_gradient * hidden_candidate
		* reset * (1.0f - reset);
	float update_logit_gradient = upstream * (previous - candidate)
		* update * (1.0f - update);

	store_f32(
		storage_buffers[push.gates_i_gradient_index], gate_base + feature,
		reset_logit_gradient
	);
	store_f32(
		storage_buffers[push.gates_i_gradient_index],
		gate_base + push.hidden_size + feature,
		update_logit_gradient
	);
	store_f32(
		storage_buffers[push.gates_i_gradient_index],
		gate_base + 2u * push.hidden_size + feature,
		candidate_logit_gradient
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index], gate_base + feature,
		reset_logit_gradient
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index],
		gate_base + push.hidden_size + feature,
		update_logit_gradient
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index],
		gate_base + 2u * push.hidden_size + feature,
		candidate_logit_gradient * reset
	);
	store_f32(
		storage_buffers[push.direct_hidden_gradient_index], index,
		upstream * update
	);
}
