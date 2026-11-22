import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint indices_index;
	uint output_index;
	uint rows;
	uint input_width;
	uint selected_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gather_last_dim")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.rows * push.selected_width) {
		return;
	}
	uint row = index / push.selected_width;
	int selected = load_i32(storage_buffers[push.indices_index], index);
	float value = selected >= 0 && uint(selected) < push.input_width
		? load_f32(storage_buffers[push.input_index], row * push.input_width + uint(selected))
		: 0.0;
	store_f32(storage_buffers[push.output_index], index, value);
}
