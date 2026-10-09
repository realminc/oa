import attributes;
import storage;

struct PushConstants {
	uint latent_index;
	uint codebook_index;
	uint indices_index;
	uint quantized_index;
	uint rows;
	uint code_dim;
	uint num_codes;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("vq_assign")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint row = dispatch_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint latent_base = row * push.code_dim;
	float best_distance = 3.402823466e+38F;
	uint best_code = 0;
	for (uint code = 0; code < push.num_codes; ++code) {
		uint code_base = code * push.code_dim;
		float distance = 0.0;
		for (uint column = 0; column < push.code_dim; ++column) {
			float difference = load_f32(storage_buffers[push.latent_index], latent_base + column)
				- load_f32(storage_buffers[push.codebook_index], code_base + column);
			distance += difference * difference;
		}
		if (distance < best_distance) {
			best_distance = distance;
			best_code = code;
		}
	}
	store_i32(storage_buffers[push.indices_index], row, int(best_code));
	uint winning_base = best_code * push.code_dim;
	for (uint column = 0; column < push.code_dim; ++column) {
		store_f32(
			storage_buffers[push.quantized_index],
			latent_base + column,
			load_f32(storage_buffers[push.codebook_index], winning_base + column));
	}
}
