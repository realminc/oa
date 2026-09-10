// Batched matrix multiplication with left-transposed storage:
// A[N,K,M]^T @ B[N,K,P] = output[N,M,P].

import storage;
import attributes;

struct PushConstants {
	uint left_index;
	uint right_index;
	uint output_index;
	uint batch;
	uint rows;
	uint inner;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("bmm_tn")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint total = push.batch * push.rows * push.columns;
	if (index >= total) {
		return;
	}
	uint matrix_size = push.rows * push.columns;
	uint batch_index = index / matrix_size;
	uint remainder = index - batch_index * matrix_size;
	uint row = remainder / push.columns;
	uint column = remainder - row * push.columns;
	uint left_base = batch_index * push.inner * push.rows + row;
	uint right_base = batch_index * push.inner * push.columns + column;
	float accumulator = 0.0f;
	for (uint k = 0; k < push.inner; ++k) {
		accumulator += load_f32(
			storage_buffers[push.left_index], left_base + k * push.rows
		) * load_f32(
			storage_buffers[push.right_index], right_base + k * push.columns
		);
	}
	store_f32(storage_buffers[push.output_index], index, accumulator);
}
