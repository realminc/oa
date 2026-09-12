import attributes;
import storage;

struct PushConstants {
	uint logits_index;
	uint output_index;
	uint rows;
	uint vocabulary;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("sample_logits")]
[domain("matrix")]
[variant("greedy")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint row = dispatch_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.vocabulary;
	float maximum = load_f32(storage_buffers[push.logits_index], base);
	uint selected = 0;
	for (uint column = 1; column < push.vocabulary; ++column) {
		float candidate = load_f32(storage_buffers[push.logits_index], base + column);
		if (candidate > maximum) {
			maximum = candidate;
			selected = column;
		}
	}
	store_i32(storage_buffers[push.output_index], row, int(selected));
}
