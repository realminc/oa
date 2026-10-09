// Rotary-position transpose adjoint adapted from OA's donor kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/rope/ropeApplyBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_gradient_index;
	uint tokens;
	uint num_heads;
	uint head_dim;
	float theta_base;
	uint position_offset;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("rope_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint half_dim = push.head_dim / 2u;
	uint pairs_per_token = push.num_heads * half_dim;
	uint index = dispatch_thread_id.x;
	if (index >= push.tokens * pairs_per_token) return;

	uint token = index / pairs_per_token;
	uint remainder = index % pairs_per_token;
	uint head = remainder / half_dim;
	uint pair = remainder % half_dim;
	uint position = token + push.position_offset;
	float frequency = float(position)
		* pow(push.theta_base, -2.0f * float(pair) / float(push.head_dim));
	float cosine = cos(frequency);
	float sine = sin(frequency);
	uint first = token * (push.num_heads * push.head_dim) + head * push.head_dim + pair;
	uint second = first + half_dim;
	float g0 = load_f32(storage_buffers[push.output_gradient_index], first);
	float g1 = load_f32(storage_buffers[push.output_gradient_index], second);
	store_f32(storage_buffers[push.input_gradient_index], first, g0 * cosine + g1 * sine);
	store_f32(storage_buffers[push.input_gradient_index], second, -g0 * sine + g1 * cosine);
}
