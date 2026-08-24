// PhiloxUniform.slang — Philox RNG Uniform Distribution [0, 1)
//
// Generates uniform random floats in [0, 1) using Philox4x32-10.
// Each thread generates 4 random floats from one Philox call.
//
// DISPATCH: (count / 1024, 1, 1) workgroups × 256 threads
// Each thread processes 4 elements (1024 elements per workgroup)

import philox;
import storage;

struct PushConstants {
	uint out_idx;    // output buffer [count] floats
	uint count;      // number of random floats to generate
	uint seedLow;    // low 32 bits of the random seed
	uint seedHigh;   // high 32 bits of the random seed
	uint offset;     // global offset for counter
	float low;       // inclusive range minimum
	float high;      // exclusive range maximum
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint global_tid = gid.x * 256u + tid.x;
	uint base_idx = global_tid * 4u;
	
	// Each thread generates 4 floats
	if (base_idx >= pc.count) return;
	
	// Counter = (global_tid, 0, 0, 0) + offset
	uint4 counter = uint4(global_tid + pc.offset, 0u, 0u, 0u);
	uint2 key = uint2(pc.seedLow, pc.seedHigh);
	
	// Generate 4 x uint32 random bits
	uint4 bits = PhiloxRng(counter, key);
	
	// Convert to floats and store
	for (uint i = 0u; i < 4u; i++) {
		uint idx = base_idx + i;
		if (idx < pc.count) {
			float r = PhiloxToFloat01(bits[i]) * (pc.high - pc.low) + pc.low;
			oaStore(heap[pc.out_idx], idx, r);
		}
	}
}
