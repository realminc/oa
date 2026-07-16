// Replay-safe Philox uniform RNG. The captured program owns one counter word
// per RNG operation; a tail dispatch advances it once after the full step.

import Philox;
import Storage;

struct PushConstants {
	uint out_idx;
	uint state_idx;
	uint count;
	uint seed;
	uint offset;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint globalTid = gid.x * 256u + tid.x;
	uint baseIdx = globalTid * 4u;
	if (baseIdx >= pc.count) return;

	uint replay = heap[pc.state_idx].Load(0u);
	uint countersPerReplay = (pc.count + 3u) / 4u;
	uint4 counter = uint4(
		globalTid + pc.offset + replay * countersPerReplay, 0u, 0u, 0u);
	uint4 bits = PhiloxRng(counter, uint2(pc.seed, 0x12345678u));

	for (uint i = 0u; i < 4u; ++i) {
		uint idx = baseIdx + i;
		if (idx < pc.count) OaStore(heap[pc.out_idx], idx, PhiloxToFloat01(bits[i]));
	}
}
