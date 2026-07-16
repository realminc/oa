import Storage;
struct PushConstants {
	uint gradOut_idx;
	uint rowMap_idx;
	uint count_idx;
	uint dispatchArgs_idx;
	uint gradSource_idx;
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
	OaStore(heap[pc.gradSource_idx], i,
		OaLoad(heap[pc.gradOut_idx], targetRow * pc.D + d));
}
