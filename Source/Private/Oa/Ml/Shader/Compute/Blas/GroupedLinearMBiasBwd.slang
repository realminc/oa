import Storage;
struct PushConstants { uint dy_idx; uint offsets_idx; uint db_idx; uint R; uint N; uint E; };
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint n = tid.x, expert = tid.y;
	if (expert >= pc.E || n >= pc.N) return;
	uint begin = heap[pc.offsets_idx].Load(expert * 4);
	uint end = heap[pc.offsets_idx].Load((expert + 1) * 4);
	float sum = 0.0;
	for (uint r = begin; r < end; ++r) sum += OaLoad(heap[pc.dy_idx], r * pc.N + n);
	OaStore(heap[pc.db_idx], expert * pc.N + n, sum);
}
