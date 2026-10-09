// Mechanical adaptation of OA SDK rlCartPoleStep.slang.
// One invocation advances one standard Gymnasium CartPole Euler lane while
// preserving the pre-action observation in separate storage.

import storage;
import attributes;

struct PushConstants {
	uint action_index;
	uint state_index;
	uint transition_observation_index;
	uint reward_index;
	uint terminated_index;
	uint truncated_index;
	uint done_index;
	uint episode_steps_index;
	uint environments;
	uint max_episode_steps;
	float gravity;
	float cart_mass;
	float pole_mass;
	float half_pole_length;
	float force_magnitude;
	float time_step;
	float position_threshold;
	float angle_threshold_radians;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("cart_pole_step")]
[domain("ml")]
[variant("gymnasium_euler")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint environment = dispatch_thread_id.x;
	if (environment >= push.environments) {
		return;
	}

	if (load_u8(storage_buffers[push.done_index], environment) != 0u) {
		store_f32(storage_buffers[push.reward_index], environment, 0.0);
		store_u8(storage_buffers[push.terminated_index], environment, 0u);
		store_u8(storage_buffers[push.truncated_index], environment, 0u);
		return;
	}

	uint state_base = environment * 4u;
	float x = load_f32(storage_buffers[push.state_index], state_base + 0u);
	float x_velocity = load_f32(storage_buffers[push.state_index], state_base + 1u);
	float angle = load_f32(storage_buffers[push.state_index], state_base + 2u);
	float angle_velocity = load_f32(storage_buffers[push.state_index], state_base + 3u);
	store_f32(storage_buffers[push.transition_observation_index], state_base + 0u, x);
	store_f32(storage_buffers[push.transition_observation_index], state_base + 1u, x_velocity);
	store_f32(storage_buffers[push.transition_observation_index], state_base + 2u, angle);
	store_f32(storage_buffers[push.transition_observation_index], state_base + 3u, angle_velocity);

	int action = load_i32(storage_buffers[push.action_index], environment);
	bool valid_action = action == 0 || action == 1;
	float force = action == 1 ? push.force_magnitude : -push.force_magnitude;
	float total_mass = push.cart_mass + push.pole_mass;
	float pole_mass_length = push.pole_mass * push.half_pole_length;
	float cosine = cos(angle);
	float sine = sin(angle);
	float temporary = (force
		+ pole_mass_length * angle_velocity * angle_velocity * sine) / total_mass;
	float angle_acceleration = (push.gravity * sine - cosine * temporary)
		/ (push.half_pole_length
			* (4.0 / 3.0 - push.pole_mass * cosine * cosine / total_mass));
	float x_acceleration = temporary
		- pole_mass_length * angle_acceleration * cosine / total_mass;

	x += push.time_step * x_velocity;
	x_velocity += push.time_step * x_acceleration;
	angle += push.time_step * angle_velocity;
	angle_velocity += push.time_step * angle_acceleration;

	store_f32(storage_buffers[push.state_index], state_base + 0u, x);
	store_f32(storage_buffers[push.state_index], state_base + 1u, x_velocity);
	store_f32(storage_buffers[push.state_index], state_base + 2u, angle);
	store_f32(storage_buffers[push.state_index], state_base + 3u, angle_velocity);

	uint episode_steps = load_u32(storage_buffers[push.episode_steps_index], environment) + 1u;
	store_u32(storage_buffers[push.episode_steps_index], environment, episode_steps);
	bool invalid_state = isnan(x) || isnan(x_velocity) || isnan(angle) || isnan(angle_velocity);
	bool terminated = !valid_action || invalid_state
		|| abs(x) > push.position_threshold
		|| abs(angle) > push.angle_threshold_radians;
	bool truncated = !terminated && episode_steps >= push.max_episode_steps;
	bool done = terminated || truncated;

	store_f32(storage_buffers[push.reward_index], environment, valid_action ? 1.0 : 0.0);
	store_u8(storage_buffers[push.terminated_index], environment, terminated ? 1u : 0u);
	store_u8(storage_buffers[push.truncated_index], environment, truncated ? 1u : 0u);
	store_u8(storage_buffers[push.done_index], environment, done ? 1u : 0u);
}
