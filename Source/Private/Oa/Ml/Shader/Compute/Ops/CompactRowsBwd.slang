import Storage;
struct PushConstants { uint gradOut_idx; uint rowMap_idx; uint count_idx; uint gradIn_idx; uint T; uint D; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.T * pc.D) return;
	uint packedRow = i / pc.D;
	uint count = heap[pc.count_idx].Load(0);
	if (packedRow >= count) return;
	uint d = i % pc.D;
	uint sourceRow = heap[pc.rowMap_idx].Load(packedRow * 4);
	OaAtomicAdd(heap[pc.gradIn_idx], sourceRow * pc.D + d, OaLoad(heap[pc.gradOut_idx], i));
}
