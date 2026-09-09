// FP32 tanh-approximate GELU used by the canonical OA Transformer.

import storage;
import attributes;

struct PushConstants { uint input_index; uint output_index; uint element_count; };
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gelu")]
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
	float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
	store_f32(storage_buffers[push.output_index], index, 0.5f * x * (1.0f + tanh(inner)));
}
