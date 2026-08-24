// Deterministic batched Lunar Lander 3D reset. One invocation owns one lane;
// completed-lane reset advances only that lane's episode counter.

static const uint CONFIG_POLICY_DT = 0u;
static const uint CONFIG_FUEL_CAPACITY = 8u;
static const uint CONFIG_POSITION_SCALE = 24u;
static const uint CONFIG_VELOCITY_SCALE = 25u;
static const uint CONFIG_ANGULAR_SCALE = 26u;
static const uint CONFIG_TERRAIN_CLEARANCE_SCALE = 27u;
static const uint CONFIG_FOOT_CLEARANCE_SCALE = 28u;
static const uint CONFIG_PROBE_SPACING = 29u;
static const uint CONFIG_TERRAIN_CELL_SIZE = 41u;
static const uint CONFIG_TERRAIN_MIN_X = 42u;
static const uint CONFIG_TERRAIN_MAX_X = 43u;
static const uint CONFIG_TERRAIN_MIN_Z = 44u;
static const uint CONFIG_TERRAIN_MAX_Z = 45u;
static const uint CONFIG_SUPPORT_BASE = 47u;

static const uint CONFIG_U32_CELLS_X = 11u;
static const uint CONFIG_U32_CELLS_Z = 12u;

static const uint STATE_F32_WIDTH = 32u;
static const uint STATE_POSITION = 0u;
static const uint STATE_LINEAR_VELOCITY = 3u;
static const uint STATE_ORIENTATION = 6u;
static const uint STATE_ANGULAR_VELOCITY = 10u;
static const uint STATE_FUEL = 13u;

static const uint STATE_U32_WIDTH = 16u;
static const uint STATE_TERMINATED = 7u;
static const uint STATE_TRUNCATED = 8u;
static const uint STATE_EPISODE_INDEX_LOW = 10u;
static const uint STATE_EPISODE_INDEX_HIGH = 11u;

static const uint OBSERVATION_WIDTH = 33u;

struct PushConstants {
	uint config_f32_idx;
	uint config_u32_idx;
	uint terrain_f32_idx;
	uint state_f32_idx;
	uint state_u32_idx;
	uint observation_idx;
	uint end_reason_idx;
	uint environments;
	uint seed_low;
	uint seed_high;
	uint only_completed;
};

struct TerrainSample {
	bool inBounds;
	float height;
	float3 normal;
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

uint2 add64(uint2 left, uint2 right) {
	uint low = left.x + right.x;
	return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

uint2 shiftRight64(uint2 value, uint shift) {
	if (shift == 0u) return value;
	if (shift < 32u) {
		return uint2(
			(value.x >> shift) | (value.y << (32u - shift)),
			value.y >> shift);
	}
	return uint2(value.y >> (shift - 32u), 0u);
}

uint2 multiply32(uint left, uint right) {
	uint leftLow = left & 0xffffu;
	uint leftHigh = left >> 16u;
	uint rightLow = right & 0xffffu;
	uint rightHigh = right >> 16u;
	uint productLow = leftLow * rightLow;
	uint productCrossA = leftLow * rightHigh;
	uint productCrossB = leftHigh * rightLow;
	uint productHigh = leftHigh * rightHigh;
	uint middle = (productLow >> 16u)
		+ (productCrossA & 0xffffu) + (productCrossB & 0xffffu);
	return uint2(
		(productLow & 0xffffu) | (middle << 16u),
		productHigh + (productCrossA >> 16u) + (productCrossB >> 16u)
			+ (middle >> 16u));
}

uint2 multiply64(uint2 left, uint2 right) {
	uint2 lowProduct = multiply32(left.x, right.x);
	return uint2(
		lowProduct.x,
		lowProduct.y + left.x * right.y + left.y * right.x);
}

uint2 mix64(uint2 value) {
	value = add64(value, uint2(0x7f4a7c15u, 0x9e3779b9u));
	value ^= shiftRight64(value, 30u);
	value = multiply64(value, uint2(0x1ce4e5b9u, 0xbf58476du));
	value ^= shiftRight64(value, 27u);
	value = multiply64(value, uint2(0x133111ebu, 0x94d049bbu));
	return value ^ shiftRight64(value, 31u);
}

uint2 combineSeed(uint2 seed, uint2 value) {
	return mix64(seed ^ mix64(value));
}

uint2 spawnSeed(
	uint environment,
	uint episodeLow,
	uint episodeHigh) {
	uint2 root = uint2(0x4e415231u, 0x4f414c55u);
	root = combineSeed(root, uint2(configU32(1u), 0u));
	root = combineSeed(root, uint2(configU32(2u), 0u));
	root = combineSeed(root, uint2(configU32(3u), 0u));
	root = combineSeed(root, uint2(configU32(4u), 0u));
	root = combineSeed(root, uint2(configU32(5u), 0u));
	root = combineSeed(root, uint2(configU32(6u), 0u));
	root = combineSeed(root, uint2(pc.seed_low, pc.seed_high));
	root = combineSeed(root, uint2(environment, 0u));
	root = combineSeed(root, uint2(episodeLow, episodeHigh));
	return combineSeed(root, uint2(0x4e303031u, 0x53504157u));
}

float sample01(
	uint environment,
	uint episodeLow,
	uint episodeHigh,
	uint component) {
	uint2 bits = combineSeed(
		spawnSeed(environment, episodeLow, episodeHigh), uint2(component, 0u));
	// This is the scalar manifest's top-53-bit conversion expressed without
	// requiring shaderInt64 storage or arithmetic capabilities. FP32 conversion
	// can round the largest UInt32 values to 2^32, so clamp to the greatest
	// representable float below one to preserve the scalar [0, 1) contract.
	float sample = float(bits.y) * 2.3283064365386963e-10
		+ float(bits.x >> 11u) * 1.1102230246251565e-16;
	return min(sample, 0.9999999403953552);
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

void writeObservation(uint environment) {
	uint stateBase = environment * STATE_F32_WIDTH;
	uint stateU32Base = environment * STATE_U32_WIDTH;
	uint observationBase = environment * OBSERVATION_WIDTH;
	float3 position = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 2u));
	float3 velocity = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 2u));
	float4 orientation = float4(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 2u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 3u));
	float3 angularVelocity = float3(
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 0u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 1u),
		loadF32(heap[pc.state_f32_idx], stateBase + STATE_ANGULAR_VELOCITY + 2u));
	float positionScale = configF32(CONFIG_POSITION_SCALE);
	float velocityScale = configF32(CONFIG_VELOCITY_SCALE);
	float angularScale = configF32(CONFIG_ANGULAR_SCALE);
	uint offset = 0u;
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(position.x, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(position.y, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(position.z, positionScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(velocity.x, velocityScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(velocity.y, velocityScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(velocity.z, velocityScale));
	float3 bodyUp = quatRotate(orientation, float3(0.0, 1.0, 0.0));
	float3 bodyForward = quatRotate(orientation, float3(0.0, 0.0, -1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.x, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.y, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyUp.z, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.x, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.y, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++, clamp(bodyForward.z, -1.0, 1.0));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(angularVelocity.x, angularScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(angularVelocity.y, angularScale));
	storeF32(heap[pc.observation_idx], observationBase + offset++,
		normalizedValue(angularVelocity.z, angularScale));
	float probeSpacing = configF32(CONFIG_PROBE_SPACING);
	float terrainScale = configF32(CONFIG_TERRAIN_CLEARANCE_SCALE);
	for (int probeZ = -1; probeZ <= 1; ++probeZ) {
		for (int probeX = -1; probeX <= 1; ++probeX) {
			TerrainSample terrain = queryTerrain(
				position.x + float(probeX) * probeSpacing,
				position.z + float(probeZ) * probeSpacing);
			float clearance = terrain.inBounds
				? position.y - terrain.height : terrainScale;
			storeF32(heap[pc.observation_idx], observationBase + offset++,
				normalizedValue(clearance, terrainScale));
		}
	}
	float footScale = configF32(CONFIG_FOOT_CLEARANCE_SCALE);
	for (uint foot = 0u; foot < 4u; ++foot) {
		uint supportIndex = 3u + foot;
		float3 center = position + quatRotate(orientation, supportOffset(supportIndex));
		TerrainSample terrain = queryTerrain(center.x, center.z);
		float clearance = terrain.inBounds
			? center.y - supportRadius(supportIndex) - terrain.height : footScale;
		storeF32(heap[pc.observation_idx], observationBase + offset++,
			normalizedValue(clearance, footScale));
	}
	uint footContacts = loadU32(heap[pc.state_u32_idx], stateU32Base + 2u);
	for (uint foot = 0u; foot < 4u; ++foot) {
		storeF32(heap[pc.observation_idx], observationBase + offset++,
			(footContacts & (1u << foot)) != 0u ? 1.0 : 0.0);
	}
	float fuel = loadF32(heap[pc.state_f32_idx], stateBase + STATE_FUEL);
	storeF32(heap[pc.observation_idx], observationBase + offset,
		clamp(fuel / configF32(CONFIG_FUEL_CAPACITY), 0.0, 1.0));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint environment = tid.x;
	if (environment >= pc.environments) return;
	uint stateU32Base = environment * STATE_U32_WIDTH;
	if (pc.only_completed != 0u
		&& loadU32(heap[pc.state_u32_idx], stateU32Base + STATE_TERMINATED) == 0u
		&& loadU32(heap[pc.state_u32_idx], stateU32Base + STATE_TRUNCATED) == 0u) {
		return;
	}
	uint episodeLow = 0u;
	uint episodeHigh = 0u;
	if (pc.only_completed != 0u) {
		episodeLow = loadU32(
			heap[pc.state_u32_idx], stateU32Base + STATE_EPISODE_INDEX_LOW) + 1u;
		episodeHigh = loadU32(
			heap[pc.state_u32_idx], stateU32Base + STATE_EPISODE_INDEX_HIGH);
		if (episodeLow == 0u) ++episodeHigh;
	}
	uint stateBase = environment * STATE_F32_WIDTH;
	for (uint index = 0u; index < STATE_F32_WIDTH; ++index) {
		storeF32(heap[pc.state_f32_idx], stateBase + index, 0.0);
	}
	for (uint index = 0u; index < STATE_U32_WIDTH; ++index) {
		storeU32(heap[pc.state_u32_idx], stateU32Base + index, 0u);
	}
	float padRange = configF32(46u) * 0.35;
	float3 position = float3(
		(sample01(environment, episodeLow, episodeHigh, 0u) * 2.0 - 1.0) * padRange,
		5.0 + sample01(environment, episodeLow, episodeHigh, 1u) * 2.0,
		(sample01(environment, episodeLow, episodeHigh, 2u) * 2.0 - 1.0) * padRange);
	float3 velocity = float3(
		(sample01(environment, episodeLow, episodeHigh, 3u) * 2.0 - 1.0) * 0.12,
		-0.1 - sample01(environment, episodeLow, episodeHigh, 4u) * 0.2,
		(sample01(environment, episodeLow, episodeHigh, 5u) * 2.0 - 1.0) * 0.12);
	float pitch = (sample01(environment, episodeLow, episodeHigh, 6u) * 2.0 - 1.0) * 0.03;
	float roll = (sample01(environment, episodeLow, episodeHigh, 7u) * 2.0 - 1.0) * 0.03;
	float yaw = (sample01(environment, episodeLow, episodeHigh, 8u) * 2.0 - 1.0) * 0.08;
	float4 yawQuat = float4(cos(yaw * 0.5), 0.0, sin(yaw * 0.5), 0.0);
	float4 pitchQuat = float4(cos(pitch * 0.5), sin(pitch * 0.5), 0.0, 0.0);
	float4 rollQuat = float4(cos(roll * 0.5), 0.0, 0.0, sin(roll * 0.5));
	float4 orientation = quatNormalize(quatMul(quatMul(yawQuat, pitchQuat), rollQuat));
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 0u, position.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 1u, position.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_POSITION + 2u, position.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 0u, velocity.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 1u, velocity.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_LINEAR_VELOCITY + 2u, velocity.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 0u, orientation.x);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 1u, orientation.y);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 2u, orientation.z);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_ORIENTATION + 3u, orientation.w);
	storeF32(heap[pc.state_f32_idx], stateBase + STATE_FUEL,
		configF32(CONFIG_FUEL_CAPACITY));
	storeU32(heap[pc.state_u32_idx],
		stateU32Base + STATE_EPISODE_INDEX_LOW, episodeLow);
	storeU32(heap[pc.state_u32_idx],
		stateU32Base + STATE_EPISODE_INDEX_HIGH, episodeHigh);
	storeU32(heap[pc.end_reason_idx], environment, 0u);
	writeObservation(environment);
}
