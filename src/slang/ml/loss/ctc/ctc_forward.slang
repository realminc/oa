// Row-cooperative FP32 connectionist temporal classification forward.
//
// One workgroup per batch item. The extended label lattice of 2*L+1
// states (blank-separated) stays in shared memory as a two-row ping-pong
// while the forward recursion sweeps time; each workgroup emits the
// item's negative log-likelihood. The host reduces the per-item losses
// into the scalar contract.

import storage;
import attributes;

struct PushConstants {
	uint log_probs_index;
	uint targets_index;
	uint input_lengths_index;
	uint target_lengths_index;
	uint nll_index;
	uint time_steps;
	uint batch;
	uint classes;
	uint target_width;
	uint blank;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint MAX_STATES = 513u;
static const float NEG_INF = -3.402823466e+38f;
static const uint PORTABLE_DISPATCH_WIDTH = 65535u;

groupshared int labels[MAX_STATES];
groupshared float alpha[2 * MAX_STATES];
groupshared uint invalid;

float log_add_exp(float a, float b) {
	float hi = max(a, b);
	return hi == NEG_INF ? NEG_INF : hi + log(exp(a - hi) + exp(b - hi));
}

[kernel_name("ctc")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint item = group_id.x + group_id.y * PORTABLE_DISPATCH_WIDTH;
	uint lane = group_thread_id.x;
	if (item >= push.batch) {
		return;
	}
	uint times = min(storage_buffers[push.input_lengths_index].Load(item * 4u), push.time_steps);
	uint length = min(storage_buffers[push.target_lengths_index].Load(item * 4u), push.target_width);
	uint states = length * 2u + 1u;
	if (lane == 0u) {
		invalid = 0u;
	}
	for (uint s = lane; s < states; s += 256u) {
		int label = (s & 1u) == 0u
			? int(push.blank)
			: int(storage_buffers[push.targets_index].Load((item * push.target_width + s / 2u) * 4u));
		labels[s] = label;
		if (label < 0 || label >= int(push.classes)) {
			invalid = 1u;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (invalid != 0u) {
		if (lane == 0u) {
			store_f32(storage_buffers[push.nll_index], item, asfloat(0x7fc00000u));
		}
		return;
	}
	if (times == 0u) {
		GroupMemoryBarrierWithGroupSync();
		if (lane == 0u) {
			store_f32(storage_buffers[push.nll_index], item, 3.402823466e+38f);
		}
		return;
	}
	GroupMemoryBarrierWithGroupSync();
	uint row = item * push.classes;
	uint plane = push.batch * push.classes;
	for (uint s = lane; s < states; s += 256u) {
		alpha[s] = s < 2u
			? load_f32(storage_buffers[push.log_probs_index], row + uint(labels[s]))
			: NEG_INF;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint t = 1u; t < times; ++t) {
		uint cur = (t & 1u) * MAX_STATES;
		uint prev = ((t + 1u) & 1u) * MAX_STATES;
		for (uint s = lane; s < states; s += 256u) {
			float best = alpha[prev + s];
			if (s > 0u) {
				best = log_add_exp(best, alpha[prev + s - 1u]);
			}
			if (s > 1u && labels[s] != int(push.blank) && labels[s] != labels[s - 2u]) {
				best = log_add_exp(best, alpha[prev + s - 2u]);
			}
			alpha[cur + s] = best + load_f32(
				storage_buffers[push.log_probs_index],
				t * plane + row + uint(labels[s])
			);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	uint last = ((times - 1u) & 1u) * MAX_STATES;
	float ll = alpha[last + states - 1u];
	if (states > 1u) {
		ll = log_add_exp(ll, alpha[last + states - 2u]);
	}
	if (lane == 0u) {
		store_f32(storage_buffers[push.nll_index], item, -ll);
	}
}
