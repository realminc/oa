// SampleSortedLogits — temperature + nucleus sampling from TopK-sorted rows.
// TopK supplies descending values and original class indices. Each invocation
// handles one inference row; no logits or probability vector crosses the host.

import Storage;

struct PushConstants {
	uint values_idx;
	uint indices_idx;
	uint random_idx;
	uint output_idx;
	uint rows;
	uint K;
	float temperature;
	float topP;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint row = tid.x;
	if (row >= pc.rows) return;
	uint base = row * pc.K;
	float maxValue = OaLoad(heap[pc.values_idx], base);
	float total = 0.0f;
	for (uint k = 0; k < pc.K; ++k)
		total += exp((OaLoad(heap[pc.values_idx], base + k) - maxValue) / pc.temperature);

	float nucleusTarget = total * pc.topP;
	float nucleusMass = 0.0f;
	for (uint k = 0; k < pc.K; ++k) {
		nucleusMass += exp((OaLoad(heap[pc.values_idx], base + k) - maxValue) / pc.temperature);
		if (nucleusMass >= nucleusTarget) break;
	}
	float threshold = OaLoadF32(heap[pc.random_idx], row) * nucleusMass;
	float cumulative = 0.0f;
	uint chosen = 0;
	for (uint k = 0; k < pc.K; ++k) {
		cumulative += exp((OaLoad(heap[pc.values_idx], base + k) - maxValue) / pc.temperature);
		chosen = k;
		if (cumulative >= threshold) break;
	}
	uint token = heap[pc.indices_idx].Load((base + chosen) * 4);
	heap[pc.output_idx].Store(row * 4, token);
}
