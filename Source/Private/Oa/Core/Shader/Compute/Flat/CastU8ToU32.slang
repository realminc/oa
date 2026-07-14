struct PushConstants {
	uint src_idx;    // bufs[0] - UInt8 buffer
	uint dst_idx;    // bufs[1] - UInt32 buffer
	uint count;      // Number of elements
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;
	
	// Load UInt8 value from a byte-addressable heap slot and cast to UInt32.
	uint byte_off = idx;
	uint word = heap[pc.src_idx].Load(byte_off & ~3u);
	uint val_u32 = (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
	
	// Store as UInt32.
	heap[pc.dst_idx].Store(idx * 4u, val_u32);
}