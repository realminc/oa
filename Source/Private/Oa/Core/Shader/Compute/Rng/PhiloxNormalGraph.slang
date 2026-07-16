// Replay-safe Philox normal RNG. See PhiloxUniformGraph.slang.

import Philox;
import Storage;

struct PushConstants {
	uint out_idx;
	uint state_idx;
	uint count;
	uint seed;
	uint offset;
	float mean;
	float stddev;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint globalTid = gid.x * 256u + tid.x;
	uint baseIdx = globalTid * 2u;
	if (baseIdx >= pc.count) return;

	uint replay = heap[pc.state_idx].Load(0u);
	uint countersPerReplay = (pc.count + 1u) / 2u;
	uint4 counter = uint4(
		globalTid + pc.offset + replay * countersPerReplay, 0u, 0u, 0u);
	uint4 bits = PhiloxRng(counter, uint2(pc.seed, 0x87654321u));
	float z0;
	float z1;
	PhiloxToNormal(bits, z0, z1);
	z0 = z0 * pc.stddev + pc.mean;
	z1 = z1 * pc.stddev + pc.mean;
	if (baseIdx < pc.count) OaStore(heap[pc.out_idx], baseIdx, z0);
	if (baseIdx + 1u < pc.count) OaStore(heap[pc.out_idx], baseIdx + 1u, z1);
}
