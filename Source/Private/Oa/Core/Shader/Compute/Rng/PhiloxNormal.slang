// PhiloxNormal.slang — Philox RNG Normal Distribution N(mean, stddev)
//
// Generates normal random floats using Philox4x32-10 + Box-Muller transform.
// Each thread generates 2 normal variates from one Philox call.
//
// DISPATCH: (count / 512, 1, 1) workgroups × 256 threads
// Each thread processes 2 elements (512 elements per workgroup)

import Philox;
import Storage;

// Bindless push constants: buffer indices come first, then user data
struct PushConstants {
	uint out_idx;    // output buffer index (heap[out_idx]) - from bindless system
	uint count;      // number of random floats to generate
	uint seed;       // random seed (key[0])
	uint offset;     // global offset for counter
	float mean;      // mean of normal distribution
	float stddev;    // standard deviation
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint global_tid = gid.x * 256u + tid.x;
	uint base_idx = global_tid * 2u;
	
	// Each thread generates 2 normal variates
	if (base_idx >= pc.count) return;
	
	// Counter = (global_tid, 0, 0, 0) + offset
	uint4 counter = uint4(global_tid + pc.offset, 0u, 0u, 0u);
	uint2 key = uint2(pc.seed, 0x87654321u);  // Fixed second key component
	
	// Generate 4 x uint32 random bits
	uint4 bits = PhiloxRng(counter, key);
	
	// Box-Muller: convert first two uint32 to two standard normals
	float z0, z1;
	PhiloxToNormal(bits, z0, z1);
	
	// Scale and shift to N(mean, stddev)
	z0 = z0 * pc.stddev + pc.mean;
	z1 = z1 * pc.stddev + pc.mean;
	
	// Store
	if (base_idx < pc.count) {
		OaStore(heap[pc.out_idx], base_idx, z0);
	}
	if (base_idx + 1u < pc.count) {
		OaStore(heap[pc.out_idx], base_idx + 1u, z1);
	}
}
