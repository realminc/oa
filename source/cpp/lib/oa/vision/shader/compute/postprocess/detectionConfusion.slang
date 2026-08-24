struct PushConstants {
	uint predicted_idx, target_idx, out_idx;
	uint count, classes;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	if (i >= pc.count) return;
	int predicted = asint(heap[pc.predicted_idx].Load(i * 4u));
	int target = asint(heap[pc.target_idx].Load(i * 4u));
	if (predicted < 0 || target < 0
		|| predicted >= int(pc.classes) || target >= int(pc.classes)) return;
	uint ignored;
	heap[pc.out_idx].InterlockedAdd(
		(uint(target) * pc.classes + uint(predicted)) * 4u, 1u, ignored);
}
