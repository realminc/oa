// Vectorized CartPole-v1 dynamics. The pre-action observation is preserved in
// a separate output before the state is advanced in place.

struct PushConstants {
	uint action_idx;
	uint state_idx;
	uint transition_observation_idx;
	uint reward_idx;
	uint terminated_idx;
	uint truncated_idx;
	uint done_idx;
	uint episode_steps_idx;
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

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(RWByteAddressBuffer buffer, uint index) {
	return asfloat(buffer.Load(index * 4u));
}

void storeF32(RWByteAddressBuffer buffer, uint index, float value) {
	buffer.Store(index * 4u, asuint(value));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint environment = tid.x;
	if (environment >= pc.environments) return;

	if (heap[pc.done_idx].Load<uint8_t>(environment) != uint8_t(0)) {
		storeF32(heap[pc.reward_idx], environment, 0.0);
		heap[pc.terminated_idx].Store<uint8_t>(environment, uint8_t(0));
		heap[pc.truncated_idx].Store<uint8_t>(environment, uint8_t(0));
		return;
	}

	uint stateBase = environment * 4u;
	float x = loadF32(heap[pc.state_idx], stateBase + 0u);
	float xVelocity = loadF32(heap[pc.state_idx], stateBase + 1u);
	float angle = loadF32(heap[pc.state_idx], stateBase + 2u);
	float angleVelocity = loadF32(heap[pc.state_idx], stateBase + 3u);
	storeF32(heap[pc.transition_observation_idx], stateBase + 0u, x);
	storeF32(heap[pc.transition_observation_idx], stateBase + 1u, xVelocity);
	storeF32(heap[pc.transition_observation_idx], stateBase + 2u, angle);
	storeF32(heap[pc.transition_observation_idx], stateBase + 3u, angleVelocity);

	int action = asint(heap[pc.action_idx].Load(environment * 4u));
	bool validAction = action == 0 || action == 1;
	float force = action == 1 ? pc.force_magnitude : -pc.force_magnitude;
	float totalMass = pc.cart_mass + pc.pole_mass;
	float poleMassLength = pc.pole_mass * pc.half_pole_length;
	float cosine = cos(angle);
	float sine = sin(angle);
	float temporary = (force
		+ poleMassLength * angleVelocity * angleVelocity * sine) / totalMass;
	float angleAcceleration = (pc.gravity * sine - cosine * temporary)
		/ (pc.half_pole_length
			* (4.0 / 3.0 - pc.pole_mass * cosine * cosine / totalMass));
	float xAcceleration = temporary
		- poleMassLength * angleAcceleration * cosine / totalMass;

	// Match Gymnasium's default Euler integrator.
	x += pc.time_step * xVelocity;
	xVelocity += pc.time_step * xAcceleration;
	angle += pc.time_step * angleVelocity;
	angleVelocity += pc.time_step * angleAcceleration;

	storeF32(heap[pc.state_idx], stateBase + 0u, x);
	storeF32(heap[pc.state_idx], stateBase + 1u, xVelocity);
	storeF32(heap[pc.state_idx], stateBase + 2u, angle);
	storeF32(heap[pc.state_idx], stateBase + 3u, angleVelocity);

	uint episodeSteps = heap[pc.episode_steps_idx].Load(environment * 4u) + 1u;
	heap[pc.episode_steps_idx].Store(environment * 4u, episodeSteps);
	bool invalidState = x != x || xVelocity != xVelocity
		|| angle != angle || angleVelocity != angleVelocity;
	bool terminated = !validAction || invalidState
		|| abs(x) > pc.position_threshold
		|| abs(angle) > pc.angle_threshold_radians;
	bool truncated = !terminated && episodeSteps >= pc.max_episode_steps;
	bool done = terminated || truncated;

	storeF32(heap[pc.reward_idx], environment, validAction ? 1.0 : 0.0);
	heap[pc.terminated_idx].Store<uint8_t>(environment,
		terminated ? uint8_t(1) : uint8_t(0));
	heap[pc.truncated_idx].Store<uint8_t>(environment,
		truncated ? uint8_t(1) : uint8_t(0));
	heap[pc.done_idx].Store<uint8_t>(environment,
		done ? uint8_t(1) : uint8_t(0));
}
