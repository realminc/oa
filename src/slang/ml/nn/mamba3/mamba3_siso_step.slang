// Mamba-3 SISO single-token recurrent step for autoregressive decoding.
// Donor: oa/ml/shader/compute/nn/ssm/mamba3/mamba3SisoStep.slang

import storage;
import activations;
import attributes;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;

struct PushConstants {
	// Bindless storage indices are prepended by the runtime in buffer order.
	uint c_idx;
	uint b_idx;
	uint x_idx;
	uint z_idx;
	uint adt_idx;
	uint dt_idx;
	uint trap_idx;
	uint angle_idx;
	uint cbias_idx;
	uint bbias_idx;
	uint d_idx;
	uint y_idx;
	uint ssm_idx;
	uint angle_state_idx;
	uint k_state_idx;
	uint v_state_idx;
	uint b;
	uint h;
	uint g;
	uint p;
	uint n;
	uint a;
	uint has_z;
	uint has_d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_c[MAXN];
groupshared float shared_b[MAXN];
groupshared float shared_theta[MAXA];
groupshared float shared_qk;

[kernel_name("mamba3_siso_step")]
[domain("ml")]
[variant("recurrent")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
	uint batch_head = group_id.x;
	uint head = batch_head % push.h;
	uint batch = batch_head / push.h;
	uint group = head / (push.h / push.g);
	uint channel = thread_id.x;
	bool active_channel = channel < push.p;

	uint vector_base = batch_head * push.p;
	uint state_vector_base = (batch * push.g + group) * push.n;
	uint ssm_base = batch_head * push.p * push.n;
	uint key_state_base = batch_head * push.n;
	uint angle_state_base = batch_head * push.a;
	float d_skip = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], head) : 0.0f;

	float dt = load_f32(storage_buffers[push.dt_idx], batch_head);
	float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], batch_head));
	float decay = exp(load_f32(storage_buffers[push.adt_idx], batch_head));
	float gamma = dt * trap;
	float deferred = dt * (1.0f - trap);

	for (uint rope = thread_id.x; rope < push.a; rope += THREADS) {
		float theta = load_f32(storage_buffers[push.angle_state_idx], angle_state_base + rope)
			+ load_f32(storage_buffers[push.angle_idx], batch * push.a + rope) * dt;
		shared_theta[rope] = theta;
		store_f32(storage_buffers[push.angle_state_idx], angle_state_base + rope, theta);
	}
	for (uint state = thread_id.x; state < push.n; state += THREADS) {
		shared_c[state] = load_f32(storage_buffers[push.c_idx], state_vector_base + state)
			+ load_f32(storage_buffers[push.cbias_idx], head * push.n + state);
		shared_b[state] = load_f32(storage_buffers[push.b_idx], state_vector_base + state)
			+ load_f32(storage_buffers[push.bbias_idx], head * push.n + state);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint rope = thread_id.x; rope < push.a; rope += THREADS) {
		float theta = shared_theta[rope];
		float cosine = cos(theta);
		float sine = sin(theta);
		uint first = 2u * rope;
		uint second = first + 1u;
		float c_first = shared_c[first];
		float c_second = shared_c[second];
		float b_first = shared_b[first];
		float b_second = shared_b[second];
		shared_c[first] = c_first * cosine - c_second * sine;
		shared_c[second] = c_first * sine + c_second * cosine;
		shared_b[first] = b_first * cosine - b_second * sine;
		shared_b[second] = b_first * sine + b_second * cosine;
	}
	GroupMemoryBarrierWithGroupSync();
	if (thread_id.x == 0u) {
		float qk = 0.0f;
		for (uint state = 0u; state < push.n; state++) {
			qk += shared_c[state] * shared_b[state];
		}
		shared_qk = qk;
	}
	GroupMemoryBarrierWithGroupSync();

	if (active_channel) {
		float x = load_f32(storage_buffers[push.x_idx], vector_base + channel);
		float previous_x = load_f32(storage_buffers[push.v_state_idx], vector_base + channel);
		float output = 0.0f;
		for (uint state = 0u; state < push.n; state++) {
			uint ssm_index = ssm_base + channel * push.n + state;
			float complete_previous = load_f32(storage_buffers[push.ssm_idx], ssm_index)
				+ previous_x
					* load_f32(storage_buffers[push.k_state_idx], key_state_base + state)
					* deferred;
			float decayed = decay * complete_previous;
			output += shared_c[state] * decayed;
			store_f32(
				storage_buffers[push.ssm_idx],
				ssm_index,
				decayed + gamma * x * shared_b[state]
			);
		}
		output += (d_skip + gamma * shared_qk) * x;
		if (push.has_z != 0u) {
			output *= silu(load_f32(storage_buffers[push.z_idx], vector_base + channel));
		}
		store_f32(storage_buffers[push.y_idx], vector_base + channel, output);
		store_f32(storage_buffers[push.v_state_idx], vector_base + channel, x);
	}

	// Every channel must finish reading the previous key before it is replaced.
	GroupMemoryBarrierWithGroupSync();
	for (uint state = thread_id.x; state < push.n; state += THREADS) {
		store_f32(storage_buffers[push.k_state_idx], key_state_base + state, shared_b[state]);
	}
}
