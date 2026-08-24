// Advance one captured RNG operation exactly once after a complete replay.

struct PushConstants {
	uint state_idx;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	uint value = heap[pc.state_idx].Load(0u);
	heap[pc.state_idx].Store(0u, value + 1u);
}
