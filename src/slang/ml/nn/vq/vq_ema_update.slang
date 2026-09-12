import attributes;
import storage;

struct PushConstants {
	uint latent_index;
	uint indices_index;
	uint embed_sum_input_index;
	uint cluster_size_input_index;
	uint codebook_input_index;
	uint embed_sum_output_index;
	uint cluster_size_output_index;
	uint codebook_output_index;
	uint rows;
	uint code_dim;
	uint num_codes;
	float decay;
	float epsilon;
	float dead_threshold;
	uint seed;
	uint normalize;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("vq_ema_update")]
[domain("ml")]
[variant("functional_state")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint code = dispatch_thread_id.x;
	if (code >= push.num_codes) {
		return;
	}
	uint count = 0;
	for (uint row = 0; row < push.rows; ++row) {
		if (load_i32(storage_buffers[push.indices_index], row) == int(code)) {
			++count;
		}
	}
	float cluster_size = push.decay
		* load_f32(storage_buffers[push.cluster_size_input_index], code)
		+ (1.0 - push.decay) * float(count);
	uint code_base = code * push.code_dim;
	float sum_of_squares = 0.0;
	for (uint column = 0; column < push.code_dim; ++column) {
		float sum = 0.0;
		for (uint row = 0; row < push.rows; ++row) {
			if (load_i32(storage_buffers[push.indices_index], row) == int(code)) {
				sum += load_f32(storage_buffers[push.latent_index], row * push.code_dim + column);
			}
		}
		float embed_sum = push.decay
			* load_f32(storage_buffers[push.embed_sum_input_index], code_base + column)
			+ (1.0 - push.decay) * sum;
		store_f32(storage_buffers[push.embed_sum_output_index], code_base + column, embed_sum);
		float value = embed_sum / max(cluster_size, push.epsilon);
		store_f32(storage_buffers[push.codebook_output_index], code_base + column, value);
		sum_of_squares += value * value;
	}
	store_f32(storage_buffers[push.cluster_size_output_index], code, cluster_size);
	if (push.normalize != 0) {
		float rms = sqrt(sum_of_squares / float(push.code_dim));
		if (rms > push.epsilon) {
			for (uint column = 0; column < push.code_dim; ++column) {
				uint index = code_base + column;
				store_f32(
					storage_buffers[push.codebook_output_index], index,
					load_f32(storage_buffers[push.codebook_output_index], index) / rms);
			}
		}
	}
	if (cluster_size < push.dead_threshold) {
		uint hash = ((code + 1) * 2654435761) ^ (push.seed * 2246822519) ^ (push.seed >> 15);
		hash ^= hash >> 13;
		hash *= 3266489917;
		hash ^= hash >> 16;
		uint revival_row = hash % push.rows;
		for (uint column = 0; column < push.code_dim; ++column) {
			float value = load_f32(
				storage_buffers[push.latent_index], revival_row * push.code_dim + column);
			store_f32(storage_buffers[push.codebook_output_index], code_base + column, value);
			store_f32(storage_buffers[push.embed_sum_output_index], code_base + column, value);
		}
		store_f32(storage_buffers[push.cluster_size_output_index], code, 1.0);
	}
}
