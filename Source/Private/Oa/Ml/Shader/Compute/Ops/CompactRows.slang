// CompactRows — stable GPU-only row compaction with fixed capacity [T,D].
// One workgroup performs a chunked prefix scan; Count and RowMap stay on-device.

import Storage;

struct PushConstants { uint x_idx; uint mask_idx; uint out_idx; uint rowMap_idx; uint count_idx; uint dispatchArgs_idx; uint T; uint D; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
groupshared uint scan[256];
groupshared uint baseOffset;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
	uint lid = tid.x;
	if (lid == 0) baseOffset = 0;
	GroupMemoryBarrierWithGroupSync();
	for (uint base = 0; base < pc.T; base += 256) {
		uint row = base + lid;
		uint selected = row < pc.T && OaLoad(heap[pc.mask_idx], row) != 0.0f ? 1 : 0;
		scan[lid] = selected;
		GroupMemoryBarrierWithGroupSync();
		for (uint offset = 1; offset < 256; offset <<= 1) {
			uint add = lid >= offset ? scan[lid - offset] : 0;
			GroupMemoryBarrierWithGroupSync();
			scan[lid] += add;
			GroupMemoryBarrierWithGroupSync();
		}
		uint slot = baseOffset + scan[lid] - selected;
		if (selected != 0) {
			heap[pc.rowMap_idx].Store(slot * 4, row);
			for (uint d = 0; d < pc.D; ++d)
				OaStore(heap[pc.out_idx], slot * pc.D + d, OaLoad(heap[pc.x_idx], row * pc.D + d));
		}
		GroupMemoryBarrierWithGroupSync();
		if (lid == 0) baseOffset += scan[255];
		GroupMemoryBarrierWithGroupSync();
	}
	if (lid == 0) {
		heap[pc.count_idx].Store(0, baseOffset);
		heap[pc.dispatchArgs_idx].Store(0, (baseOffset * pc.D + 255) / 256);
		heap[pc.dispatchArgs_idx].Store(4, 1);
		heap[pc.dispatchArgs_idx].Store(8, 1);
	}
}
