import attributes;
import storage;

struct PushConstants {
	uint gradient_output_index;
	uint indices_index;
	uint gradient_input_index;
	uint rows;
	uint input_width;
	uint selected_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gather_last_dim_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.rows * push.input_width) {
		return;
	}
	uint row = index / push.input_width;
	uint column = index % push.input_width;
	float sum = 0.0;
	for (uint selected = 0; selected < push.selected_width; ++selected) {
		uint route = row * push.selected_width + selected;
		int source = load_i32(storage_buffers[push.indices_index], route);
		if (source >= 0 && uint(source) == column) {
			sum += load_f32(storage_buffers[push.gradient_output_index], route);
		}
	}
	store_f32(storage_buffers[push.gradient_input_index], index, sum);
}
