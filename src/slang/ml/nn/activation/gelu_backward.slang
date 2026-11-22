// Complete adjoint of the tanh-approximate FP32 GELU activation.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint element_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gelu_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) return;
	float x = load_f32(storage_buffers[push.input_index], index);
	float x2 = x * x;
	float inner = 0.7978845608028654f * (x + 0.044715f * x2 * x);
	float t = tanh(inner);
	float derivative = 0.5f * (1.0f + t)
		+ 0.5f * x * (1.0f - t * t) * 0.7978845608028654f * (1.0f + 0.134145f * x2);
	float output_gradient = load_f32(storage_buffers[push.output_gradient_index], index);
	store_f32(storage_buffers[push.input_gradient_index], index, output_gradient * derivative);
}
