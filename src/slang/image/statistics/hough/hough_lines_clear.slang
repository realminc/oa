import attributes;
import storage;

struct PushConstants {
	uint acc_index, lines_index;
	uint acc_elements, line_elements;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("hough_lines_clear")]
[domain("image")]
[variant("replay_safe_zero")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index < push.acc_elements) {
		store_u32(storage_buffers[push.acc_index], index, 0u);
	} else if (index < push.acc_elements + push.line_elements) {
		store_u32(storage_buffers[push.lines_index], index - push.acc_elements, 0u);
	}
}
