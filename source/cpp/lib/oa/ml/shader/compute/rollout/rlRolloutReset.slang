// Clears the validity mask without reallocating rollout storage.

struct PushConstants {
	uint valid_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint index = tid.x;
	if (index >= pc.count) return;
	heap[pc.valid_idx].Store<uint8_t>(index, uint8_t(0));
}
