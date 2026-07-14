import Storage;

struct PushConstants {
	uint gradOut_idx;
	uint indices_idx;
	uint gradIn_idx;
	uint T;
	uint E;
	uint K;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.T * pc.E) return;
	uint t = i / pc.E;
	uint e = i % pc.E;
	float sum = 0.0;
	for (uint k = 0; k < pc.K; ++k) {
		uint route = t * pc.K + k;
		int selected = asint(heap[pc.indices_idx].Load(route * 4));
		if (selected >= 0 && uint(selected) == e)
			sum += OaLoad(heap[pc.gradOut_idx], route);
	}
	OaStore(heap[pc.gradIn_idx], i, sum);
}
