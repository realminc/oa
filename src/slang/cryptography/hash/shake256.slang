import attributes;
import keccak;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint message_len;
	uint num_blocks;
	uint squeeze_u64;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint load_message_byte(uint row_base, uint byte_index) {
	uint address = row_base + byte_index;
	uint word = storage_buffers[push.input_index].Load(address & ~3u);
	return (word >> ((address & 3u) * 8u)) & 0xffu;
}

uint2 load_padded_lane(uint row_base, uint padded_byte, uint rate_bytes) {
	uint2 lane = uint2(0, 0);
	for (uint byte = 0; byte < 8; ++byte) {
		uint position = padded_byte + byte;
		uint value = position < push.message_len ? load_message_byte(row_base, position) : 0;
		if (position == push.message_len) value |= 0x1fu;
		if (position == push.num_blocks * rate_bytes - 1) value |= 0x80u;
		if (byte < 4) lane.x |= value << (byte * 8);
		else lane.y |= value << ((byte - 4) * 8);
	}
	return lane;
}

[kernel_name("shake256")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) return;
	static const uint RATE_LANES = 17;
	static const uint RATE_BYTES = 136;
	uint2 state[25];
	for (uint lane = 0; lane < 25; ++lane) state[lane] = uint2(0, 0);
	uint row_base = index * push.message_len;
	for (uint block = 0; block < push.num_blocks; ++block) {
		for (uint lane = 0; lane < RATE_LANES; ++lane) {
			state[lane] ^= load_padded_lane(row_base, block * RATE_BYTES + lane * 8, RATE_BYTES);
		}
		keccak_f1600_u32(state);
	}
	uint output_base = index * push.squeeze_u64 * 8;
	uint remaining = push.squeeze_u64;
	uint offset = 0;
	while (remaining > 0) {
		uint amount = min(remaining, RATE_LANES);
		for (uint lane = 0; lane < amount; ++lane) {
			uint address = output_base + (offset + lane) * 8;
			storage_buffers[push.output_index].Store<uint>(address, state[lane].x);
			storage_buffers[push.output_index].Store<uint>(address + 4, state[lane].y);
		}
		offset += amount;
		remaining -= amount;
		if (remaining > 0) keccak_f1600_u32(state);
	}
}
