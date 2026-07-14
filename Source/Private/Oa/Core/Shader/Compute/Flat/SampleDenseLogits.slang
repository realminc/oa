// SampleDenseLogits — exact temperature sampling without top-k/top-p sorting.
// Streaming three-pass implementation supports arbitrary vocabulary sizes and
// reads back only one chosen class ID per row.

import Storage;

struct PushConstants {
	uint logits_idx;
	uint random_idx;
	uint output_idx;
	uint rows;
	uint vocab;
	float temperature;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint row = tid.x;
	if (row >= pc.rows) return;
	uint base = row * pc.vocab;
	float maxValue = OaLoad(heap[pc.logits_idx], base);
	for (uint k = 1; k < pc.vocab; ++k) maxValue = max(maxValue, OaLoad(heap[pc.logits_idx], base + k));
	float total = 0.0f;
	for (uint k = 0; k < pc.vocab; ++k)
		total += exp((OaLoad(heap[pc.logits_idx], base + k) - maxValue) / pc.temperature);
	float threshold = OaLoadF32(heap[pc.random_idx], row) * total;
	float cumulative = 0.0f;
	uint chosen = 0;
	for (uint k = 0; k < pc.vocab; ++k) {
		cumulative += exp((OaLoad(heap[pc.logits_idx], base + k) - maxValue) / pc.temperature);
		chosen = k;
		if (cumulative >= threshold) break;
	}
	heap[pc.output_idx].Store(row * 4, chosen);
}
