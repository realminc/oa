import Storage;

struct PushConstants {
	uint self_idx;
	uint indices_idx;
	uint out_idx;
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
	if (i >= pc.T * pc.K) return;
	uint t = i / pc.K;
	int e = asint(heap[pc.indices_idx].Load(i * 4));
	float value = (e >= 0 && uint(e) < pc.E)
		? OaLoad(heap[pc.self_idx], t * pc.E + uint(e)) : 0.0;
	OaStore(heap[pc.out_idx], i, value);
}
