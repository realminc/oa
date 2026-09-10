// BatchNorm2d input adjoint from saved statistics and channel reductions.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint mean_index;
	uint variance_index;
	uint weight_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint input_gradient_index;
	uint element_count;
	uint channels;
	uint height;
	uint width;
	float epsilon;
	uint training;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("batch_norm_2d_input_backward")]
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
	uint spatial = push.height * push.width;
	uint channel = (index / spatial) % push.channels;
	float variance = load_f32(storage_buffers[push.variance_index], channel);
	float inverse_stddev = rsqrt(max(variance, 0.0f) + push.epsilon);
	float weight = load_f32(storage_buffers[push.weight_index], channel);
	float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
	float input_gradient = output_gradient * weight * inverse_stddev;
	if (push.training != 0u) {
		uint sample_count = push.element_count / push.channels;
		float input = load_f32(storage_buffers[push.input_index], index);
		float mean = load_f32(storage_buffers[push.mean_index], channel);
		float normalized = (input - mean) * inverse_stddev;
		float weight_gradient = load_f32(storage_buffers[push.weight_gradient_index], channel);
		float bias_gradient = load_f32(storage_buffers[push.bias_gradient_index], channel);
		input_gradient = weight * inverse_stddev * (
			output_gradient
			- bias_gradient / float(sample_count)
			- normalized * weight_gradient / float(sample_count)
		);
	}
	store_f32(storage_buffers[push.input_gradient_index], index, input_gradient);
}
