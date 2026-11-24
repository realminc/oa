// Row-cooperative FP32 connectionist temporal classification adjoint.
//
// One workgroup per batch item. Phase A recomputes the item's forward
// lattice into a host-owned workspace; phase B sweeps the backward
// recursion from T-1 to 0, emitting the per-class posterior
// -exp(alpha_t(s)+beta_t(s)-ll) / batch into the log-probability
// gradient through fixed-point-free CAS accumulation.

import storage;
import attributes;

struct PushConstants {
	uint log_probs_index;
	uint targets_index;
	uint input_lengths_index;
	uint target_lengths_index;
	uint alpha_index;
	uint gradient_index;
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
groupshared float beta[2 * MAX_STATES];
groupshared uint invalid;

float log_add_exp(float a, float b) {
	float hi = max(a, b);
	return hi == NEG_INF ? NEG_INF : hi + log(exp(a - hi) + exp(b - hi));
}

[kernel_name("ctc_backward")]
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
	uint state_stride = push.target_width * 2u + 1u;
	uint alpha_base = item * push.time_steps * state_stride;
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
	if (times == 0u || invalid != 0u) {
		return;
	}
	uint row = item * push.classes;
	uint plane = push.batch * push.classes;
	// Phase A: forward lattice into the workspace.
	for (uint s = lane; s < states; s += 256u) {
		float value = s < 2u
			? load_f32(storage_buffers[push.log_probs_index], row + uint(labels[s]))
			: NEG_INF;
		alpha[s] = value;
		store_f32(storage_buffers[push.alpha_index], alpha_base + s, value);
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
			float value = best + load_f32(
				storage_buffers[push.log_probs_index],
				t * plane + row + uint(labels[s])
			);
			alpha[cur + s] = value;
			store_f32(storage_buffers[push.alpha_index], alpha_base + t * state_stride + s, value);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	AllMemoryBarrierWithGroupSync();
	uint last = ((times - 1u) & 1u) * MAX_STATES;
	float ll = alpha[last + states - 1u];
	if (states > 1u) {
		ll = log_add_exp(ll, alpha[last + states - 2u]);
	}
	float inv_batch = 1.0f / float(push.batch);
	// Phase B: backward sweep; posterior at every step.
	for (uint t = times; t-- > 0u;) {
		uint cur = (t & 1u) * MAX_STATES;
		if (t == times - 1u) {
			for (uint s = lane; s < states; s += 256u) {
				bool terminal = s == states - 1u || (states > 1u && s == states - 2u);
				beta[cur + s] = terminal ? 0.0f : NEG_INF;
			}
		} else {
			uint prev = ((t + 1u) & 1u) * MAX_STATES;
			for (uint s = lane; s < states; s += 256u) {
				float best = beta[prev + s] + load_f32(
					storage_buffers[push.log_probs_index],
					(t + 1u) * plane + row + uint(labels[s])
				);
				if (s + 1u < states) {
					best = log_add_exp(best, beta[prev + s + 1u] + load_f32(
						storage_buffers[push.log_probs_index],
						(t + 1u) * plane + row + uint(labels[s + 1u])
					));
				}
				if (s + 2u < states && labels[s] != int(push.blank) && labels[s] != labels[s + 2u]) {
					best = log_add_exp(best, beta[prev + s + 2u] + load_f32(
						storage_buffers[push.log_probs_index],
						(t + 1u) * plane + row + uint(labels[s + 2u])
					));
				}
				beta[cur + s] = best;
			}
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint s = lane; s < states; s += 256u) {
			float a = load_f32(storage_buffers[push.alpha_index], alpha_base + t * state_stride + s);
			float posterior = exp(a + beta[cur + s] - ll);
			if (posterior > 0.0f) {
				atomic_add_f32(
					storage_buffers[push.gradient_index],
					t * plane + row + uint(labels[s]),
					-posterior * inv_batch
				);
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
