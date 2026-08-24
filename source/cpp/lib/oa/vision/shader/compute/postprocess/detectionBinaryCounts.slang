struct PushConstants {
	uint predicted_idx, target_idx, out_idx;
	uint count;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.count) return;
	bool predicted = heap[pc.predicted_idx].Load<uint8_t>(i) != 0;
	bool target = heap[pc.target_idx].Load<uint8_t>(i) != 0;
	uint bucket = predicted ? (target ? 0u : 1u) : (target ? 2u : 3u);
	uint ignored;
	heap[pc.out_idx].InterlockedAdd(bucket * 4u, 1u, ignored);
}
