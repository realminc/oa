// Advance mutable AdamW replay state once per complete optimizer step.
// A separately recorded one-thread dispatch keeps every parameter update in
// the step on the same bias-correction index while allowing command-buffer
// replay to progress without a host-side step push constant.

struct PushConstants {
	uint state_idx;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	uint previous;
	heap[pc.state_idx].InterlockedAdd(0, 1, previous);
}
