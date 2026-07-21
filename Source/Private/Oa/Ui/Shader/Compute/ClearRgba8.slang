// ClearRgba8 — write one packed RGBA8 value to a buffer-backed texture.

struct PushConstants {
	uint out_idx;
	uint count;
	uint rgba;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	const uint idx = tid.x;
	if (idx < pc.count) {
		heap[pc.out_idx].Store(idx * 4, pc.rgba);
	}
}
