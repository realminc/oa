import attributes;
import storage;

struct PushConstants {
	uint codebook_index;
	uint indices_index;
	uint output_index;
	uint index_count;
	uint num_codes;
	uint code_dim;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("vq_lookup")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.index_count * push.code_dim) {
		return;
	}
	uint row = index / push.code_dim;
	uint column = index % push.code_dim;
	int code = load_i32(storage_buffers[push.indices_index], row);
	float value = code >= 0 && uint(code) < push.num_codes
		? load_f32(storage_buffers[push.codebook_index], uint(code) * push.code_dim + column)
		: 0.0;
	store_f32(storage_buffers[push.output_index], index, value);
}
