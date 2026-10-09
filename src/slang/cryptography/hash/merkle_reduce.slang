import attributes;
import keccak;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint hash_bytes;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("merkle_reduce")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) return;
	uint2 state[25];
	for (uint lane = 0; lane < 25; ++lane) state[lane] = uint2(0, 0);
	uint children_base = index * push.hash_bytes * 2;
	for (uint lane = 0; lane < 8; ++lane) {
		uint address = children_base + lane * 8;
		state[lane] = uint2(
			storage_buffers[push.input_index].Load<uint>(address),
			storage_buffers[push.input_index].Load<uint>(address + 4)
		);
	}
	state[8].x ^= 0x1fu;
	state[16].y ^= 0x80000000u;
	keccak_f1600_u32(state);
	uint output_base = index * push.hash_bytes;
	for (uint lane = 0; lane < 4; ++lane) {
		uint address = output_base + lane * 8;
		storage_buffers[push.output_index].Store<uint>(address, state[lane].x);
		storage_buffers[push.output_index].Store<uint>(address + 4, state[lane].y);
	}
}
