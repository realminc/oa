// Pointwise portion of one LSTM cell adjoint.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint gates_i_index;
	uint gates_h_index;
	uint hidden_index;
	uint cell_index;
	uint hidden_gradient_index;
	uint cell_gradient_index;
	uint gates_i_gradient_index;
	uint gates_h_gradient_index;
	uint direct_hidden_gradient_index;
	uint direct_cell_gradient_index;
	uint hidden_size;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("lstm_cell_backward")]
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
	uint gate_base = batch_index * (4u * push.hidden_size);

	float input_i = load_f32(storage_buffers[push.gates_i_index], gate_base + feature);
	float input_f = load_f32(
		storage_buffers[push.gates_i_index], gate_base + push.hidden_size + feature
	);
	float input_g = load_f32(
		storage_buffers[push.gates_i_index], gate_base + 2u * push.hidden_size + feature
	);
	float input_o = load_f32(
		storage_buffers[push.gates_i_index], gate_base + 3u * push.hidden_size + feature
	);
	float hidden_i = load_f32(storage_buffers[push.gates_h_index], gate_base + feature);
	float hidden_f = load_f32(
		storage_buffers[push.gates_h_index], gate_base + push.hidden_size + feature
	);
	float hidden_g = load_f32(
		storage_buffers[push.gates_h_index], gate_base + 2u * push.hidden_size + feature
	);
	float hidden_o = load_f32(
		storage_buffers[push.gates_h_index], gate_base + 3u * push.hidden_size + feature
	);
	float previous = load_f32(storage_buffers[push.cell_index], index);
	float input_gate = sigmoid(input_i + hidden_i);
	float forget_gate = sigmoid(input_f + hidden_f);
	float candidate = tanh(input_g + hidden_g);
	float output_gate = sigmoid(input_o + hidden_o);
	float cell_next = forget_gate * previous + input_gate * candidate;
	float tanh_cell = tanh(cell_next);
	float upstream = load_f32(storage_buffers[push.hidden_gradient_index], index);
	float cell_upstream = load_f32(storage_buffers[push.cell_gradient_index], index)
		+ upstream * output_gate * (1.0f - tanh_cell * tanh_cell);

	float input_pre = cell_upstream * candidate * input_gate * (1.0f - input_gate);
	float forget_pre = cell_upstream * previous * forget_gate * (1.0f - forget_gate);
	float candidate_pre = cell_upstream * input_gate * (1.0f - candidate * candidate);
	float output_pre = upstream * tanh_cell * output_gate * (1.0f - output_gate);

	store_f32(
		storage_buffers[push.gates_i_gradient_index], gate_base + feature,
		input_pre
	);
	store_f32(
		storage_buffers[push.gates_i_gradient_index],
		gate_base + push.hidden_size + feature,
		forget_pre
	);
	store_f32(
		storage_buffers[push.gates_i_gradient_index],
		gate_base + 2u * push.hidden_size + feature,
		candidate_pre
	);
	store_f32(
		storage_buffers[push.gates_i_gradient_index],
		gate_base + 3u * push.hidden_size + feature,
		output_pre
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index], gate_base + feature,
		input_pre
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index],
		gate_base + push.hidden_size + feature,
		forget_pre
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index],
		gate_base + 2u * push.hidden_size + feature,
		candidate_pre
	);
	store_f32(
		storage_buffers[push.gates_h_gradient_index],
		gate_base + 3u * push.hidden_size + feature,
		output_pre
	);
	store_f32(storage_buffers[push.direct_hidden_gradient_index], index, 0.0f);
	store_f32(
		storage_buffers[push.direct_cell_gradient_index], index,
		cell_upstream * forget_gate
	);
}
