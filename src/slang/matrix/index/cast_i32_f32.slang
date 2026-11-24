import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
	uint source_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("cast_i32_f32")]
[domain("matrix")]
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
	uint raw = storage_buffers[push.input_index].Load(index * 4u);
	float value = push.source_dtype == 0u ? float(asint(raw)) : float(raw);
	store_f32(storage_buffers[push.output_index], index, value);
}
