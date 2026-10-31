// OA donor: source/cpp/lib/oa/crypto/shader/compute/merkle/merkleReduce.slang.
// Odd-level policy: source/cpp/lib/oa/crypto/hash.cpp, merkleRoot/buildMerkleTree.
// Adaptation: existing paired-word permutation, checked ABI, duplicate-last tail.
// One invocation owns a complete 32-byte parent; no packed-word write overlap.
import attributes;
import merkle_pair;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint count;
	uint hash_bytes;
	uint leaf_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("merkle_reduce")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.count) return;
	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.hash_bytes != 32u || push.leaf_count == 0u
		|| push.count != push.leaf_count / 2u + (push.leaf_count & 1u)
		|| push.leaf_count > input_extent / 32u || push.count > output_extent / 32u) return;
	uint left = index * 2u;
	uint right = min(left + 1u, push.leaf_count - 1u);
	uint current[8];
	for (uint word = 0u; word < 8u; ++word)
		current[word] = storage_buffers[push.input_index].Load(left * 32u + word * 4u);
	merkle_hash_pair(current, storage_buffers[push.input_index], right * 32u, false);
	uint output_base = index * 32u;
	for (uint word = 0u; word < 8u; ++word)
		storage_buffers[push.output_index].Store<uint>(output_base + word * 4u, current[word]);
}
