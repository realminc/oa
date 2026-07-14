import Storage;
struct PushConstants { uint self_idx; uint source_idx; uint rowMap_idx; uint count_idx; uint out_idx; uint T; uint D; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.T * pc.D) return;
	uint row = i / pc.D;
	uint d = i % pc.D;
	uint count = heap[pc.count_idx].Load(0);
	uint lo = 0, hi = count;
	while (lo < hi) {
		uint mid = (lo + hi) >> 1;
		if (heap[pc.rowMap_idx].Load(mid * 4) < row) lo = mid + 1; else hi = mid;
	}
	float value = OaLoad(heap[pc.self_idx], i);
	if (lo < count && heap[pc.rowMap_idx].Load(lo * 4) == row)
		value += OaLoad(heap[pc.source_idx], lo * pc.D + d);
	OaStore(heap[pc.out_idx], i, value);
}
