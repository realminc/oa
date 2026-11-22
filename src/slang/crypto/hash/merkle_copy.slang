import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("merkle_copy")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) return;
	uint address = index * 4;
	storage_buffers[push.output_index].Store<uint>(
		address,
		storage_buffers[push.input_index].Load<uint>(address)
	);
}
