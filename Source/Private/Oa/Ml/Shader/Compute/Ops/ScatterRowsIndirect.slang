// GPU-driven sparse row add. CompactRows authored the exact workgroup count;
// zero selected rows therefore becomes a legal zero-work indirect dispatch.
import Storage;
struct PushConstants {
	uint source_idx;
	uint rowMap_idx;
	uint count_idx;
	uint dispatchArgs_idx;
	uint out_idx;
	uint T;
	uint D;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	uint count = heap[pc.count_idx].Load(0);
	if (i >= count * pc.D) return;
	uint packedRow = i / pc.D;
	uint d = i % pc.D;
	uint targetRow = heap[pc.rowMap_idx].Load(packedRow * 4);
	uint target = targetRow * pc.D + d;
	float value = OaLoad(heap[pc.out_idx], target)
		+ OaLoad(heap[pc.source_idx], i);
	OaStore(heap[pc.out_idx], target, value);
}
