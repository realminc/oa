// FP32 batched Lunar Lander 3D dynamics. One invocation advances one lane in
// fixed order, so no atomics or inter-lane synchronization affect simulation.

static const uint CONFIG_POLICY_DT = 0u;
static const uint CONFIG_GRAVITY = 1u;
static const uint CONFIG_MASS = 2u;
static const uint CONFIG_INERTIA_X = 3u;
static const uint CONFIG_INERTIA_Y = 4u;
static const uint CONFIG_INERTIA_Z = 5u;
static const uint CONFIG_MAIN_THRUST = 6u;
static const uint CONFIG_ATTITUDE_TORQUE = 7u;
static const uint CONFIG_FUEL_CAPACITY = 8u;
static const uint CONFIG_MAIN_FUEL_RATE = 9u;
static const uint CONFIG_ATTITUDE_FUEL_RATE = 10u;
static const uint CONFIG_RESTITUTION = 11u;
static const uint CONFIG_FRICTION = 12u;
static const uint CONFIG_CONTACT_SLOP = 13u;
static const uint CONFIG_CORRECTION_FRACTION = 14u;
static const uint CONFIG_MAX_POSITION_CORRECTION = 15u;
static const uint CONFIG_MAX_CONTACT_IMPULSE = 16u;
static const uint CONFIG_MAX_BIAS_SPEED = 17u;
static const uint CONFIG_TASK_MINIMUM_Y = 18u;
static const uint CONFIG_TASK_MAXIMUM_Y = 19u;
static const uint CONFIG_SAFE_LINEAR_SPEED = 20u;
static const uint CONFIG_SAFE_ANGULAR_SPEED = 21u;
static const uint CONFIG_SAFE_TILT = 22u;
static const uint CONFIG_HARD_FOOT_SPEED = 23u;
static const uint CONFIG_POSITION_SCALE = 24u;
static const uint CONFIG_VELOCITY_SCALE = 25u;
static const uint CONFIG_ANGULAR_SCALE = 26u;
static const uint CONFIG_TERRAIN_CLEARANCE_SCALE = 27u;
static const uint CONFIG_FOOT_CLEARANCE_SCALE = 28u;
static const uint CONFIG_PROBE_SPACING = 29u;
static const uint CONFIG_REWARD_GAMMA = 30u;
static const uint CONFIG_POSITION_POTENTIAL_WEIGHT = 31u;
static const uint CONFIG_VELOCITY_POTENTIAL_WEIGHT = 32u;
static const uint CONFIG_TILT_POTENTIAL_WEIGHT = 33u;
static const uint CONFIG_ANGULAR_POTENTIAL_WEIGHT = 34u;
static const uint CONFIG_MAIN_FUEL_COST_WEIGHT = 35u;
static const uint CONFIG_ATTITUDE_FUEL_COST_WEIGHT = 36u;
static const uint CONFIG_SOFT_FOOT_REWARD = 37u;
static const uint CONFIG_STABLE_DWELL_REWARD = 38u;
static const uint CONFIG_SUCCESS_REWARD = 39u;
static const uint CONFIG_FAILURE_PENALTY = 40u;
static const uint CONFIG_TERRAIN_CELL_SIZE = 41u;
static const uint CONFIG_TERRAIN_MIN_X = 42u;
static const uint CONFIG_TERRAIN_MAX_X = 43u;
static const uint CONFIG_TERRAIN_MIN_Z = 44u;
static const uint CONFIG_TERRAIN_MAX_Z = 45u;
static const uint CONFIG_PAD_HALF_EXTENT = 46u;
static const uint CONFIG_SUPPORT_BASE = 47u;

static const uint CONFIG_U32_PHYSICS_SUBSTEPS = 7u;
static const uint CONFIG_U32_CONTACT_ITERATIONS = 8u;
static const uint CONFIG_U32_SAFE_DWELL_STEPS = 9u;
static const uint CONFIG_U32_MAX_EPISODE_STEPS = 10u;
static const uint CONFIG_U32_CELLS_X = 11u;
static const uint CONFIG_U32_CELLS_Z = 12u;

static const uint STATE_F32_WIDTH = 32u;
static const uint STATE_POSITION = 0u;
static const uint STATE_LINEAR_VELOCITY = 3u;
static const uint STATE_ORIENTATION = 6u;
static const uint STATE_ANGULAR_VELOCITY = 10u;
static const uint STATE_FUEL = 13u;
static const uint STATE_MAIN_THROTTLE = 14u;
static const uint STATE_ATTITUDE_COMMAND = 15u;
static const uint STATE_BODY_IMPULSES = 18u;
static const uint STATE_FOOT_IMPULSES = 21u;
static const uint STATE_EPISODE_RETURN = 25u;

static const uint STATE_U32_WIDTH = 16u;
static const uint STATE_LAST_ACTION = 0u;
static const uint STATE_BODY_CONTACTS = 1u;
static const uint STATE_FOOT_CONTACTS = 2u;
static const uint STATE_FEET_ON_PAD = 3u;
static const uint STATE_FOOT_REWARDED = 4u;
static const uint STATE_EPISODE_STEP = 5u;
static const uint STATE_STABLE_DWELL = 6u;
static const uint STATE_TERMINATED = 7u;
static const uint STATE_TRUNCATED = 8u;
static const uint STATE_END_REASON = 9u;

static const uint END_NONE = 0u;
static const uint END_SAFE_LANDING = 1u;
static const uint END_BODY_IMPACT = 2u;
static const uint END_HARD_FOOT_IMPACT = 3u;
static const uint END_OUT_OF_BOUNDS = 4u;
static const uint END_NUMERICAL_FAILURE = 5u;
static const uint END_TIME_LIMIT = 6u;
static const uint END_EXTERNAL_STOP = 7u;
static const uint END_INVALID_ACTION = 8u;

static const uint OBSERVATION_WIDTH = 33u;

struct PushConstants {
	uint action_idx;
	uint external_stop_idx;
	uint config_f32_idx;
	uint config_u32_idx;
	uint terrain_f32_idx;
	uint state_f32_idx;
	uint state_u32_idx;
	uint transition_observation_idx;
	uint observation_idx;
	uint reward_idx;
	uint terminated_idx;
	uint truncated_idx;
	uint end_reason_idx;
	uint environments;
};

struct TerrainSample {
	bool inBounds;
	float height;
	float3 normal;
};

struct LanderState {
	float3 position;
	float3 linearVelocity;
	float4 orientation;
	float3 angularVelocity;
	float fuel;
	float mainThrottle;
	float3 attitudeCommand;
	float bodyImpulses[3];
	float footImpulses[4];
	float episodeReturn;
	uint lastAction;
	uint bodyContacts;
	uint footContacts;
	uint feetOnPad;
	uint footRewarded;
	uint episodeStep;
	uint stableDwell;
	uint terminated;
	uint truncated;
	uint endReason;
};

struct ContactDiagnostics {
	float maximumPenetration;
	float maximumNormalImpulse;
	float maximumFrictionImpulse;
	float maximumFootClosingSpeed;
	float totalPositionCorrection;
	uint contactCount;
	bool bodyContactOccurred;
	bool footContactOccurred;
	bool bounded;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(RWByteAddressBuffer buffer, uint index) {
	return asfloat(buffer.Load(index * 4u));
}

void storeF32(RWByteAddressBuffer buffer, uint index, float value) {
	buffer.Store(index * 4u, asuint(value));
}

uint loadU32(RWByteAddressBuffer buffer, uint index) {
	return buffer.Load(index * 4u);
}

void storeU32(RWByteAddressBuffer buffer, uint index, uint value) {
	buffer.Store(index * 4u, value);
}

float configF32(uint index) {
	return loadF32(heap[pc.config_f32_idx], index);
}

uint configU32(uint index) {
	return loadU32(heap[pc.config_u32_idx], index);
}

float3 inverseInertia() {
	return float3(
		1.0 / configF32(CONFIG_INERTIA_X),
		1.0 / configF32(CONFIG_INERTIA_Y),
		1.0 / configF32(CONFIG_INERTIA_Z));
}

float4 quatMul(float4 left, float4 right) {
	return float4(
		left.x * right.x - left.y * right.y - left.z * right.z
			- left.w * right.w,
		left.x * right.y + left.y * right.x + left.z * right.w
			- left.w * right.z,
		left.x * right.z - left.y * right.w + left.z * right.x
			+ left.w * right.y,
		left.x * right.w + left.y * right.z - left.z * right.y
			+ left.w * right.x);
}

float4 quatNormalize(float4 orientation) {
	float lengthSquared = dot(orientation, orientation);
	return lengthSquared > 1.0e-20
		? orientation * rsqrt(lengthSquared) : float4(1.0, 0.0, 0.0, 0.0);
}

float3 quatRotate(float4 orientation, float3 value) {
	float4 normalized = quatNormalize(orientation);
	float3 imaginary = normalized.yzw;
	float3 twiceCross = 2.0 * cross(imaginary, value);
	return value + normalized.x * twiceCross + cross(imaginary, twiceCross);
}

float3 quatInverseRotate(float4 orientation, float3 value) {
	float4 normalized = quatNormalize(orientation);
	return quatRotate(
		float4(normalized.x, -normalized.y, -normalized.z, -normalized.w), value);
}

bool finiteState(LanderState state) {
	float orientationNormSquared = dot(state.orientation, state.orientation);
	bool impulsesFinite = true;
	for (uint index = 0u; index < 3u; ++index) {
		impulsesFinite = impulsesFinite && isfinite(state.bodyImpulses[index]);
	}
	for (uint index = 0u; index < 4u; ++index) {
		impulsesFinite = impulsesFinite && isfinite(state.footImpulses[index]);
	}
	return all(isfinite(state.position)) && all(isfinite(state.linearVelocity))
		&& all(isfinite(state.orientation)) && isfinite(orientationNormSquared)
		&& orientationNormSquared > 1.0e-20
		&& all(isfinite(state.angularVelocity)) && isfinite(state.fuel)
		&& state.fuel >= 0.0 && state.fuel <= configF32(CONFIG_FUEL_CAPACITY)
		&& isfinite(state.mainThrottle) && all(isfinite(state.attitudeCommand))
		&& impulsesFinite && isfinite(state.episodeReturn);
}

float normalizedValue(float value, float scale) {
	return clamp(value / scale, -1.0, 1.0);
}

TerrainSample queryTerrain(float positionX, float positionZ) {
	TerrainSample sample;
	sample.inBounds = false;
	sample.height = 0.0;
	sample.normal = float3(0.0, 1.0, 0.0);
	float minimumX = configF32(CONFIG_TERRAIN_MIN_X);
	float maximumX = configF32(CONFIG_TERRAIN_MAX_X);
	float minimumZ = configF32(CONFIG_TERRAIN_MIN_Z);
	float maximumZ = configF32(CONFIG_TERRAIN_MAX_Z);
	if (positionX < minimumX || positionX > maximumX
		|| positionZ < minimumZ || positionZ > maximumZ) {
		return sample;
	}
	uint cellsX = configU32(CONFIG_U32_CELLS_X);
	uint cellsZ = configU32(CONFIG_U32_CELLS_Z);
	float cellSize = configF32(CONFIG_TERRAIN_CELL_SIZE);
	float gridX = (positionX - minimumX) / cellSize;
	float gridZ = (positionZ - minimumZ) / cellSize;
	uint cellX = gridX >= float(cellsX)
		? cellsX - 1u : uint(floor(gridX));
	uint cellZ = gridZ >= float(cellsZ)
		? cellsZ - 1u : uint(floor(gridZ));
	float localX = gridX >= float(cellsX) ? 1.0 : gridX - float(cellX);
	float localZ = gridZ >= float(cellsZ) ? 1.0 : gridZ - float(cellZ);
	uint row = cellsX + 1u;
	uint index00 = cellZ * row + cellX;
	float height00 = loadF32(heap[pc.terrain_f32_idx], index00);
	float height10 = loadF32(heap[pc.terrain_f32_idx], index00 + 1u);
	float height01 = loadF32(heap[pc.terrain_f32_idx], index00 + row);
	float height11 = loadF32(heap[pc.terrain_f32_idx], index00 + row + 1u);
	float heightDx;
	float heightDz;
	if (localZ <= localX) {
		sample.height = height00 + localX * (height10 - height00)
			+ localZ * (height11 - height10);
		heightDx = (height10 - height00) / cellSize;
		heightDz = (height11 - height10) / cellSize;
	} else {
		sample.height = height00 + localX * (height11 - height01)
			+ localZ * (height01 - height00);
		heightDx = (height11 - height01) / cellSize;
		heightDz = (height01 - height00) / cellSize;
	}
	sample.normal = normalize(float3(-heightDx, 1.0, -heightDz));
	sample.inBounds = true;
	return sample;
}

float3 supportOffset(uint supportIndex) {
	uint base = CONFIG_SUPPORT_BASE + supportIndex * 4u;
	return float3(configF32(base), configF32(base + 1u), configF32(base + 2u));
}

float supportRadius(uint supportIndex) {
	return configF32(CONFIG_SUPPORT_BASE + supportIndex * 4u + 3u);
}

LanderState loadState(uint environment) {
	LanderState state;
	uint stateBase = environment * STATE_F32_WIDTH;
	uint stateU32Base = environment * STATE_U32_WIDTH;
	state.position = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 2u));
	state.linearVelocity = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 2u));
	state.orientation = float4(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 2u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 3u));
	state.angularVelocity = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 2u));
	state.fuel = loadF32(heap[pc.state_f32_idx], stateBase + STATE_FUEL);
	state.mainThrottle = loadF32(
		heap[pc.state_f32_idx], stateBase + STATE_MAIN_THROTTLE);
	state.attitudeCommand = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 2u));
	for (uint index = 0u; index < 3u; ++index) {
		state.bodyImpulses[index] = loadF32(
			heap[pc.state_f32_idx], stateBase + STATE_BODY_IMPULSES + index);
	}
	for (uint index = 0u; index < 4u; ++index) {
		state.footImpulses[index] = loadF32(
			heap[pc.state_f32_idx], stateBase + STATE_FOOT_IMPULSES + index);
	}
	state.episodeReturn = loadF32(
		heap[pc.state_f32_idx], stateBase + STATE_EPISODE_RETURN);
	state.lastAction = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_LAST_ACTION);
	state.bodyContacts = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_BODY_CONTACTS);
	state.footContacts = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_FOOT_CONTACTS);
	state.feetOnPad = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_FEET_ON_PAD);
	state.footRewarded = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_FOOT_REWARDED);
	state.episodeStep = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_EPISODE_STEP);
	state.stableDwell = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_STABLE_DWELL);
	state.terminated = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_TERMINATED);
	state.truncated = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_TRUNCATED);
	state.endReason = loadU32(
		heap[pc.state_u32_idx], stateU32Base + STATE_END_REASON);
	return state;
}

void storeState(uint environment, LanderState state) {
	uint stateBase = environment * STATE_F32_WIDTH;
	uint stateU32Base = environment * STATE_U32_WIDTH;
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 0u, state.position.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 1u, state.position.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 2u, state.position.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 0u, state.linearVelocity.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 1u, state.linearVelocity.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 2u, state.linearVelocity.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 0u, state.orientation.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 1u, state.orientation.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 2u, state.orientation.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 3u, state.orientation.w);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 0u, state.angularVelocity.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 1u, state.angularVelocity.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 2u, state.angularVelocity.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_FUEL, state.fuel);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_MAIN_THROTTLE, state.mainThrottle);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 0u, state.attitudeCommand.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 1u, state.attitudeCommand.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ATTITUDE_COMMAND + 2u, state.attitudeCommand.z);
	for (uint index = 0u; index < 3u; ++index) {
		storeF32(heap[pc.state_f32_idx],
			stateBase + STATE_BODY_IMPULSES + index, state.bodyImpulses[index]);
	}
	for (uint index = 0u; index < 4u; ++index) {
		storeF32(heap[pc.state_f32_idx],
			stateBase + STATE_FOOT_IMPULSES + index, state.footImpulses[index]);
	}
	storeF32(heap[pc.state_f32_idx],
		stateBase + STATE_EPISODE_RETURN, state.episodeReturn);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_LAST_ACTION, state.lastAction);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_BODY_CONTACTS, state.bodyContacts);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_FOOT_CONTACTS, state.footContacts);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_FEET_ON_PAD, state.feetOnPad);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_FOOT_REWARDED, state.footRewarded);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_EPISODE_STEP, state.episodeStep);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_STABLE_DWELL, state.stableDwell);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_TERMINATED, state.terminated);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_TRUNCATED, state.truncated);
	storeU32(heap[pc.state_u32_idx], stateU32Base + STATE_END_REASON, state.endReason);
}

float effectiveInverseMass(
	LanderState state,
	float3 leverWorld,
	float3 directionWorld) {
	float3 rotationalWorld = cross(leverWorld, directionWorld);
	float3 rotationalBody = quatInverseRotate(state.orientation, rotationalWorld);
	return 1.0 / configF32(CONFIG_MASS)
		+ dot(rotationalBody, rotationalBody * inverseInertia());
}

void applyImpulse(
	inout LanderState state,
	float3 leverWorld,
	float3 impulseWorld) {
	state.linearVelocity += impulseWorld / configF32(CONFIG_MASS);
	float3 angularImpulseWorld = cross(leverWorld, impulseWorld);
	float3 angularImpulseBody = quatInverseRotate(
		state.orientation, angularImpulseWorld);
	state.angularVelocity += angularImpulseBody * inverseInertia();
}

void resolveSupport(
	uint supportIndex,
	bool isFoot,
	float substepTime,
	inout LanderState state,
	inout ContactDiagnostics diagnostics) {
	float3 center = state.position + quatRotate(
		state.orientation, supportOffset(supportIndex));
	TerrainSample terrain = queryTerrain(center.x, center.z);
	if (!terrain.inBounds) return;
	float separation = center.y - supportRadius(supportIndex) - terrain.height;
	if (separation >= 0.0) return;
	float penetration = -separation;
	diagnostics.maximumPenetration = max(
		diagnostics.maximumPenetration, penetration);
	++diagnostics.contactCount;
	float3 leverWorld = center - state.position;
	float3 angularVelocityWorld = quatRotate(
		state.orientation, state.angularVelocity);
	float3 pointVelocity = state.linearVelocity
		+ cross(angularVelocityWorld, leverWorld);
	float normalVelocity = dot(pointVelocity, terrain.normal);
	float closingSpeed = max(0.0, -normalVelocity);
	if (isFoot) {
		diagnostics.footContactOccurred = true;
		diagnostics.maximumFootClosingSpeed = max(
			diagnostics.maximumFootClosingSpeed, closingSpeed);
	}
	float biasSpeed = min(
		configF32(CONFIG_MAX_BIAS_SPEED),
		penetration * configF32(CONFIG_CORRECTION_FRACTION) / substepTime);
	float targetDeltaSpeed = max(
		0.0, -(1.0 + configF32(CONFIG_RESTITUTION)) * normalVelocity
			+ biasSpeed);
	float normalImpulse = targetDeltaSpeed / effectiveInverseMass(
		state, leverWorld, terrain.normal);
	normalImpulse = clamp(
		normalImpulse, 0.0, configF32(CONFIG_MAX_CONTACT_IMPULSE));
	applyImpulse(state, leverWorld, terrain.normal * normalImpulse);
	diagnostics.maximumNormalImpulse = max(
		diagnostics.maximumNormalImpulse, normalImpulse);

	angularVelocityWorld = quatRotate(state.orientation, state.angularVelocity);
	pointVelocity = state.linearVelocity + cross(angularVelocityWorld, leverWorld);
	float3 tangentVelocity = pointVelocity
		- terrain.normal * dot(pointVelocity, terrain.normal);
	float tangentSpeed = length(tangentVelocity);
	float frictionImpulse = 0.0;
	if (tangentSpeed > 1.0e-12) {
		float3 tangentDirection = -tangentVelocity / tangentSpeed;
		frictionImpulse = min(
			tangentSpeed / effectiveInverseMass(
				state, leverWorld, tangentDirection),
			configF32(CONFIG_FRICTION) * normalImpulse);
		frictionImpulse = clamp(
			frictionImpulse, 0.0, configF32(CONFIG_MAX_CONTACT_IMPULSE));
		applyImpulse(state, leverWorld, tangentDirection * frictionImpulse);
	}
	diagnostics.maximumFrictionImpulse = max(
		diagnostics.maximumFrictionImpulse, frictionImpulse);
	float correction = min(
		configF32(CONFIG_MAX_POSITION_CORRECTION),
		penetration * configF32(CONFIG_CORRECTION_FRACTION));
	state.position += terrain.normal * correction;
	diagnostics.totalPositionCorrection += correction;
	if (isFoot) {
		uint foot = supportIndex - 3u;
		state.footContacts |= 1u << foot;
		state.footImpulses[foot] += normalImpulse;
	} else {
		diagnostics.bodyContactOccurred = true;
		state.bodyContacts |= 1u << supportIndex;
		state.bodyImpulses[supportIndex] += normalImpulse;
	}
}

void refreshContacts(inout LanderState state) {
	state.bodyContacts = 0u;
	state.footContacts = 0u;
	state.feetOnPad = 0u;
	for (uint supportIndex = 0u; supportIndex < 7u; ++supportIndex) {
		float3 center = state.position + quatRotate(
			state.orientation, supportOffset(supportIndex));
		TerrainSample terrain = queryTerrain(center.x, center.z);
		bool contact = terrain.inBounds
			&& center.y - supportRadius(supportIndex) - terrain.height
				<= configF32(CONFIG_CONTACT_SLOP);
		if (supportIndex < 3u) {
			if (contact) state.bodyContacts |= 1u << supportIndex;
		} else {
			uint foot = supportIndex - 3u;
			if (contact) state.footContacts |= 1u << foot;
			if (terrain.inBounds
				&& max(abs(center.x), abs(center.z))
					<= configF32(CONFIG_PAD_HALF_EXTENT)) {
				state.feetOnPad |= 1u << foot;
			}
		}
	}
}

float3 actionTorque(int action) {
	float magnitude = configF32(CONFIG_ATTITUDE_TORQUE);
	if (action == 2) return float3(magnitude, 0.0, 0.0);
	if (action == 3) return float3(-magnitude, 0.0, 0.0);
	if (action == 4) return float3(0.0, 0.0, magnitude);
	if (action == 5) return float3(0.0, 0.0, -magnitude);
	if (action == 6) return float3(0.0, magnitude, 0.0);
	if (action == 7) return float3(0.0, -magnitude, 0.0);
	return float3(0.0, 0.0, 0.0);
}

float potential(LanderState state) {
	if (!finiteState(state)) return 0.0;
	float positionCost = min(
		1.0, length(state.position) / configF32(CONFIG_POSITION_SCALE));
	float velocityCost = min(
		1.0, length(state.linearVelocity) / configF32(CONFIG_VELOCITY_SCALE));
	float3 bodyUp = quatRotate(state.orientation, float3(0.0, 1.0, 0.0));
	float tiltCost = clamp((1.0 - bodyUp.y) * 0.5, 0.0, 1.0);
	float angularCost = min(
		1.0, length(state.angularVelocity) / configF32(CONFIG_ANGULAR_SCALE));
	return -(
		configF32(CONFIG_POSITION_POTENTIAL_WEIGHT) * positionCost
		+ configF32(CONFIG_VELOCITY_POTENTIAL_WEIGHT) * velocityCost
		+ configF32(CONFIG_TILT_POTENTIAL_WEIGHT) * tiltCost
		+ configF32(CONFIG_ANGULAR_POTENTIAL_WEIGHT) * angularCost);
}

bool instantaneouslySafe(LanderState state) {
	float3 bodyUp = quatRotate(state.orientation, float3(0.0, 1.0, 0.0));
	float tilt = acos(clamp(bodyUp.y, -1.0, 1.0));
	return (state.footContacts & 0x0fu) == 0x0fu
		&& (state.feetOnPad & 0x0fu) == 0x0fu
		&& state.bodyContacts == 0u
		&& length(state.linearVelocity) <= configF32(CONFIG_SAFE_LINEAR_SPEED)
		&& length(state.angularVelocity) <= configF32(CONFIG_SAFE_ANGULAR_SPEED)
		&& tilt <= configF32(CONFIG_SAFE_TILT);
}

void writeObservation(uint environment) {
	LanderState state = loadState(environment);
	uint observationBase = environment * OBSERVATION_WIDTH;
	uint offset = 0u;
	if (!finiteState(state)) {
		for (uint index = 0u; index < OBSERVATION_WIDTH; ++index) {
			storeF32(heap[pc.observation_idx], observationBase + index, 0.0);
		}
		return;
	}
	float positionScale = configF32(CONFIG_POSITION_SCALE);
	float velocityScale = configF32(CONFIG_VELOCITY_SCALE);
	float angularScale = configF32(CONFIG_ANGULAR_SCALE);
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.position.x, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.position.y, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.position.z, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.linearVelocity.x, velocityScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.linearVelocity.y, velocityScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.linearVelocity.z, velocityScale));
	float3 bodyUp = quatRotate(state.orientation, float3(0.0, 1.0, 0.0));
	float3 bodyForward = quatRotate(state.orientation, float3(0.0, 0.0, -1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.x, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.y, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.z, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.x, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.y, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.z, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.angularVelocity.x, angularScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.angularVelocity.y, angularScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(state.angularVelocity.z, angularScale));
	float probeSpacing = configF32(CONFIG_PROBE_SPACING);
	float terrainScale = configF32(CONFIG_TERRAIN_CLEARANCE_SCALE);
	for (int probeZ = -1; probeZ <= 1; ++probeZ) {
		for (int probeX = -1; probeX <= 1; ++probeX) {
			TerrainSample terrain = queryTerrain(
				state.position.x + float(probeX) * probeSpacing,
				state.position.z + float(probeZ) * probeSpacing);
			float clearance = terrain.inBounds
				? state.position.y - terrain.height : terrainScale;
			storeF32(heap[pc.observation_idx], observationBase + offset++,
				normalizedValue(clearance, terrainScale));
		}
	}
	float footScale = configF32(CONFIG_FOOT_CLEARANCE_SCALE);
	for (uint foot = 0u; foot < 4u; ++foot) {
		uint supportIndex = 3u + foot;
		float3 center = state.position + quatRotate(
			state.orientation, supportOffset(supportIndex));
		TerrainSample terrain = queryTerrain(center.x, center.z);
		float clearance = terrain.inBounds
			? center.y - supportRadius(supportIndex) - terrain.height : footScale;
		storeF32(heap[pc.observation_idx], observationBase + offset++,
			normalizedValue(clearance, footScale));
	}
	for (uint foot = 0u; foot < 4u; ++foot) {
		storeF32(heap[pc.observation_idx], observationBase + offset++,
			(state.footContacts & (1u << foot)) != 0u ? 1.0 : 0.0);
	}
	storeF32(heap[pc.observation_idx], observationBase + offset,
		clamp(state.fuel / configF32(CONFIG_FUEL_CAPACITY), 0.0, 1.0));
}

void copyObservation(uint environment) {
	uint base = environment * OBSERVATION_WIDTH;
	for (uint index = 0u; index < OBSERVATION_WIDTH; ++index) {
		storeF32(heap[pc.transition_observation_idx], base + index,
			loadF32(heap[pc.observation_idx], base + index));
	}
}

void writeTransition(
	uint environment,
	float reward,
	uint terminated,
	uint truncated,
	uint endReason) {
	storeF32(heap[pc.reward_idx], environment, reward);
	heap[pc.terminated_idx].Store<uint8_t>(
		environment, terminated != 0u ? uint8_t(1) : uint8_t(0));
	heap[pc.truncated_idx].Store<uint8_t>(
		environment, truncated != 0u ? uint8_t(1) : uint8_t(0));
	storeU32(heap[pc.end_reason_idx], environment, endReason);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint environment = tid.x;
	if (environment >= pc.environments) return;
	copyObservation(environment);
	LanderState state = loadState(environment);
	if (state.terminated != 0u || state.truncated != 0u) {
		writeTransition(
			environment, 0.0, state.terminated, state.truncated, state.endReason);
		return;
	}

	int action = asint(heap[pc.action_idx].Load(environment * 4u));
	if (action < 0 || action >= 8) {
		++state.episodeStep;
		state.terminated = 1u;
		state.endReason = END_INVALID_ACTION;
		float failurePenalty = configF32(CONFIG_FAILURE_PENALTY);
		state.episodeReturn += failurePenalty;
		storeState(environment, state);
		writeObservation(environment);
		writeTransition(
			environment, failurePenalty, 1u, 0u, END_INVALID_ACTION);
		return;
	}
	if (heap[pc.external_stop_idx].Load<uint8_t>(environment) != uint8_t(0)) {
		state.truncated = 1u;
		state.endReason = END_EXTERNAL_STOP;
		storeState(environment, state);
		writeObservation(environment);
		writeTransition(environment, 0.0, 0u, 1u, END_EXTERNAL_STOP);
		return;
	}

	float potentialBefore = potential(state);
	state.lastAction = uint(action);
	state.mainThrottle = 0.0;
	state.attitudeCommand = float3(0.0, 0.0, 0.0);
	state.bodyContacts = 0u;
	state.footContacts = 0u;
	state.feetOnPad = 0u;
	for (uint index = 0u; index < 3u; ++index) state.bodyImpulses[index] = 0.0;
	for (uint index = 0u; index < 4u; ++index) state.footImpulses[index] = 0.0;
	ContactDiagnostics diagnostics;
	diagnostics.maximumPenetration = 0.0;
	diagnostics.maximumNormalImpulse = 0.0;
	diagnostics.maximumFrictionImpulse = 0.0;
	diagnostics.maximumFootClosingSpeed = 0.0;
	diagnostics.totalPositionCorrection = 0.0;
	diagnostics.contactCount = 0u;
	diagnostics.bodyContactOccurred = false;
	diagnostics.footContactOccurred = false;
	diagnostics.bounded = true;
	float mainFuelUsed = 0.0;
	float attitudeFuelUsed = 0.0;
	bool physicsValid = finiteState(state);
	uint substeps = configU32(CONFIG_U32_PHYSICS_SUBSTEPS);
	uint contactIterations = configU32(CONFIG_U32_CONTACT_ITERATIONS);
	float substepTime = configF32(CONFIG_POLICY_DT) / float(substeps);
	for (uint substep = 0u; substep < substeps && physicsValid; ++substep) {
		float mainActivation = 0.0;
		float3 attitudeTorque = float3(0.0, 0.0, 0.0);
		if (action == 1) {
			float requestedFuel = configF32(CONFIG_MAIN_FUEL_RATE) * substepTime;
			mainActivation = requestedFuel > 0.0
				? min(1.0, state.fuel / requestedFuel) : 1.0;
			float fuelUsed = requestedFuel * mainActivation;
			state.fuel -= fuelUsed;
			mainFuelUsed += fuelUsed;
		} else if (action != 0) {
			float requestedFuel = configF32(CONFIG_ATTITUDE_FUEL_RATE) * substepTime;
			float activation = requestedFuel > 0.0
				? min(1.0, state.fuel / requestedFuel) : 1.0;
			float fuelUsed = requestedFuel * activation;
			state.fuel -= fuelUsed;
			attitudeFuelUsed += fuelUsed;
			attitudeTorque = actionTorque(action) * activation;
		}
		state.fuel = max(0.0, state.fuel);
		state.mainThrottle = mainActivation;
		state.attitudeCommand = attitudeTorque;
		float3 thrustWorld = quatRotate(
			state.orientation,
			float3(0.0, configF32(CONFIG_MAIN_THRUST) * mainActivation, 0.0));
		float3 linearAcceleration = thrustWorld / configF32(CONFIG_MASS)
			+ float3(0.0, -configF32(CONFIG_GRAVITY), 0.0);
		state.linearVelocity += linearAcceleration * substepTime;
		state.position += state.linearVelocity * substepTime;
		float3 inertia = float3(
			configF32(CONFIG_INERTIA_X),
			configF32(CONFIG_INERTIA_Y),
			configF32(CONFIG_INERTIA_Z));
		float3 angularMomentum = inertia * state.angularVelocity;
		float3 gyroscopicTorque = cross(state.angularVelocity, angularMomentum);
		float3 angularAcceleration = (attitudeTorque - gyroscopicTorque)
			* inverseInertia();
		state.angularVelocity += angularAcceleration * substepTime;
		float4 derivative = quatMul(
			state.orientation,
			float4(
				0.0,
				state.angularVelocity.x,
				state.angularVelocity.y,
				state.angularVelocity.z)) * 0.5;
		state.orientation = quatNormalize(
			state.orientation + derivative * substepTime);
		physicsValid = finiteState(state);
		for (uint iteration = 0u;
			iteration < contactIterations && physicsValid;
			++iteration) {
			for (uint supportIndex = 0u; supportIndex < 3u; ++supportIndex) {
				resolveSupport(
					supportIndex, false, substepTime, state, diagnostics);
			}
			for (uint supportIndex = 3u; supportIndex < 7u; ++supportIndex) {
				resolveSupport(
					supportIndex, true, substepTime, state, diagnostics);
			}
			physicsValid = finiteState(state)
				&& isfinite(diagnostics.maximumPenetration)
				&& isfinite(diagnostics.maximumNormalImpulse)
				&& isfinite(diagnostics.maximumFrictionImpulse)
				&& isfinite(diagnostics.maximumFootClosingSpeed)
				&& isfinite(diagnostics.totalPositionCorrection);
		}
	}
	if (physicsValid) refreshContacts(state);
	float maximumCorrection = float(substeps * contactIterations * 7u)
		* configF32(CONFIG_MAX_POSITION_CORRECTION);
	diagnostics.bounded = diagnostics.maximumNormalImpulse
			<= configF32(CONFIG_MAX_CONTACT_IMPULSE)
		&& diagnostics.maximumFrictionImpulse
			<= configF32(CONFIG_MAX_CONTACT_IMPULSE)
		&& diagnostics.totalPositionCorrection <= maximumCorrection + 1.0e-6;
	++state.episodeStep;
	bool instantaneousSafe = false;
	if (!physicsValid || !finiteState(state)) {
		state.terminated = 1u;
		state.endReason = END_NUMERICAL_FAILURE;
	} else if (!queryTerrain(state.position.x, state.position.z).inBounds
		|| state.position.y < configF32(CONFIG_TASK_MINIMUM_Y)
		|| state.position.y > configF32(CONFIG_TASK_MAXIMUM_Y)) {
		state.terminated = 1u;
		state.endReason = END_OUT_OF_BOUNDS;
	} else if (diagnostics.bodyContactOccurred || state.bodyContacts != 0u) {
		state.terminated = 1u;
		state.endReason = END_BODY_IMPACT;
	} else if (diagnostics.maximumFootClosingSpeed
		> configF32(CONFIG_HARD_FOOT_SPEED)) {
		state.terminated = 1u;
		state.endReason = END_HARD_FOOT_IMPACT;
	} else {
		instantaneousSafe = instantaneouslySafe(state);
		state.stableDwell = instantaneousSafe ? state.stableDwell + 1u : 0u;
		if (state.stableDwell >= configU32(CONFIG_U32_SAFE_DWELL_STEPS)) {
			state.terminated = 1u;
			state.endReason = END_SAFE_LANDING;
		}
	}
	if (state.terminated == 0u
		&& state.episodeStep >= configU32(CONFIG_U32_MAX_EPISODE_STEPS)) {
		state.truncated = 1u;
		state.endReason = END_TIME_LIMIT;
	}

	float potentialAfter = potential(state);
	float effectivePotentialAfter = state.terminated != 0u ? 0.0 : potentialAfter;
	float shaping = configF32(CONFIG_REWARD_GAMMA) * effectivePotentialAfter
		- potentialBefore;
	float mainFuelCost = -configF32(CONFIG_MAIN_FUEL_COST_WEIGHT) * mainFuelUsed;
	float attitudeFuelCost = -configF32(CONFIG_ATTITUDE_FUEL_COST_WEIGHT)
		* attitudeFuelUsed;
	float softFootContact = 0.0;
	if (diagnostics.maximumFootClosingSpeed <= configF32(CONFIG_HARD_FOOT_SPEED)) {
		for (uint foot = 0u; foot < 4u; ++foot) {
			uint mask = 1u << foot;
			if ((state.footContacts & mask) != 0u
				&& (state.footRewarded & mask) == 0u) {
				softFootContact += configF32(CONFIG_SOFT_FOOT_REWARD);
				state.footRewarded |= mask;
			}
		}
	}
	float stableDwellReward = instantaneousSafe
		? configF32(CONFIG_STABLE_DWELL_REWARD) : 0.0;
	float terminalReward = 0.0;
	if (state.endReason == END_SAFE_LANDING) {
		terminalReward = configF32(CONFIG_SUCCESS_REWARD);
	} else if (state.terminated != 0u) {
		terminalReward = configF32(CONFIG_FAILURE_PENALTY);
	}
	float reward = shaping + mainFuelCost + attitudeFuelCost
		+ softFootContact + stableDwellReward + terminalReward;
	float nextEpisodeReturn = state.episodeReturn + reward;
	if (!isfinite(reward) || !isfinite(nextEpisodeReturn)) {
		// Configuration admission bounds every normal reward accumulation. This
		// remains a device-side containment edge for arithmetic that became
		// non-finite because the physical state failed numerically.
		state.terminated = 1u;
		state.truncated = 0u;
		state.endReason = END_NUMERICAL_FAILURE;
		reward = configF32(CONFIG_FAILURE_PENALTY);
		nextEpisodeReturn = state.episodeReturn + reward;
		if (!isfinite(nextEpisodeReturn)) nextEpisodeReturn = 0.0;
	}
	state.episodeReturn = nextEpisodeReturn;
	storeState(environment, state);
	writeObservation(environment);
	writeTransition(
		environment, reward, state.terminated, state.truncated, state.endReason);
}
