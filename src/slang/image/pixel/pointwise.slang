import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index, element_count, operation;
	float p0, p1, p2, p3;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("pointwise")]
[domain("image")]
[variant("parameterized")]
[dtype("f32")]
[layout("dense")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.element_count) return;
	float value = load_f32(storage_buffers[push.input_index], index);
	float result = value;
	switch (push.operation) {
	case 0u: result = value > push.p0 ? push.p1 : 0.0f; break;
	case 1u: result = value > push.p0 ? 0.0f : push.p1; break;
	case 2u: result = min(value, push.p0); break;
	case 3u: result = value > push.p0 ? value : 0.0f; break;
	case 4u: result = value > push.p0 ? 0.0f : value; break;
	case 5u: result = value >= push.p0 && value <= push.p1 ? push.p2 : 0.0f; break;
	case 6u: result = clamp(value, push.p0, push.p1); break;
	case 7u: result = push.p0 - value; break;
	case 8u: result = value * push.p0 + push.p1; break;
	case 9u: result = pow(max(value, 0.0f), push.p0) * push.p1; break;
	case 10u: result = value >= push.p0 ? push.p1 - value : value; break;
	case 11u: {
		float levels = max(push.p0, 2.0f);
		float normalized = clamp((value - push.p1) / (push.p2 - push.p1), 0.0f, 1.0f);
		result = floor(normalized * (levels - 1.0f) + 0.5f) / (levels - 1.0f);
		result = result * (push.p2 - push.p1) + push.p1;
		break;
	}
	default: break;
	}
	store_f32(storage_buffers[push.output_index], index, result);
}
