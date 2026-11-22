// Reference-faithful single-token Mamba-3 MIMO decode.
// The [P,N] SSM state is shared across rank channels. K/V caches retain every
// rank so the deferred trapezoid contribution can complete the previous state.

import storage;
import activations;
import attributes;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;
static const uint MAXR = 8u;
static const uint MAXP = 128u;

struct PushConstants {
	uint c_idx, b_idx, x_idx, z_idx, adt_idx, dt_idx, trap_idx, angle_idx;
	uint cbias_idx, bbias_idx, d_idx, mx_idx, mz_idx, mo_idx, norm_idx, y_idx;
	uint ssm_idx, angle_state_idx, k_state_idx, v_state_idx;
	uint b, h, g, p, n, a, r;
	uint has_z, has_d, has_norm;
	float eps;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float gsQ[MAXR * MAXN];
groupshared float gsK[MAXR * MAXN];
groupshared float gsTheta[MAXA];
groupshared float gsPre[MAXR * MAXP];
groupshared float gsInv[MAXR];

[kernel_name("mamba3_mimo_step")]
[domain("ml")]
[variant("recurrent")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint g = h / (push.h / push.g);
	uint p = tid.x;
	bool activeP = p < push.p;
	uint scalar = b * push.h + h;
	float dt = load_f32(storage_buffers[push.dt_idx], scalar);
	float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], scalar));
	float decay = exp(load_f32(storage_buffers[push.adt_idx], scalar));
	float gamma = dt * trap;
	float deferred = dt * (1.0f - trap);

	uint angleInput = b * push.a;
	uint angleState = (b * push.h + h) * push.a;
	for (uint a = tid.x; a < push.a; a += THREADS) {
		float theta = load_f32(storage_buffers[push.angle_state_idx], angleState + a)
			+ load_f32(storage_buffers[push.angle_idx], angleInput + a) * dt;
		gsTheta[a] = theta;
		store_f32(storage_buffers[push.angle_state_idx], angleState + a, theta);
	}
	for (uint linear = tid.x; linear < push.r * push.n; linear += THREADS) {
		uint r = linear / push.n;
		uint n = linear % push.n;
		uint rg = r * push.g + g;
		uint input = ((b * (push.r * push.g) + rg) * push.n) + n;
		uint bias = (h * push.r + r) * push.n + n;
		gsQ[linear] = load_f32(storage_buffers[push.c_idx], input) + load_f32(storage_buffers[push.cbias_idx], bias);
		gsK[linear] = load_f32(storage_buffers[push.b_idx], input) + load_f32(storage_buffers[push.bbias_idx], bias);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint linear = tid.x; linear < push.r * push.a; linear += THREADS) {
		uint r = linear / push.a;
		uint a = linear % push.a;
		uint i0 = r * push.n + 2u * a;
		uint i1 = i0 + 1u;
		float cs = cos(gsTheta[a]), sn = sin(gsTheta[a]);
		float q0 = gsQ[i0], q1 = gsQ[i1];
		float k0 = gsK[i0], k1 = gsK[i1];
		gsQ[i0] = q0 * cs - q1 * sn;
		gsQ[i1] = q0 * sn + q1 * cs;
		gsK[i0] = k0 * cs - k1 * sn;
		gsK[i1] = k0 * sn + k1 * cs;
	}
	GroupMemoryBarrierWithGroupSync();

	if (activeP) {
		uint xp = (b * push.h + h) * push.p + p;
		uint stateBase = (b * push.h + h) * push.p * push.n + p * push.n;
		float xv = load_f32(storage_buffers[push.x_idx], xp);
		float u[MAXN];
		for (uint n = 0u; n < push.n; ++n) {
			float complete = load_f32(storage_buffers[push.ssm_idx], stateBase + n);
			float currentU = 0.0f;
			for (uint r = 0u; r < push.r; ++r) {
				uint rankState = ((b * push.h + h) * push.r + r);
				float prevV = load_f32(storage_buffers[push.v_state_idx], rankState * push.p + p);
				float prevK = load_f32(storage_buffers[push.k_state_idx], rankState * push.n + n);
				complete += deferred * prevV * prevK;
				uint rp = (h * push.r + r) * push.p + p;
				currentU += xv * load_f32(storage_buffers[push.mx_idx], rp) * gsK[r * push.n + n];
			}
			float hd = decay * complete;
			u[n] = currentU;
			store_f32(storage_buffers[push.ssm_idx], stateBase + n, hd + gamma * currentU);
		}
		float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;
		for (uint r = 0u; r < push.r; ++r) {
			float pre = 0.0f;
			for (uint n = 0u; n < push.n; ++n) {
				float stateNow = load_f32(storage_buffers[push.ssm_idx], stateBase + n);
				pre += gsQ[r * push.n + n] * stateNow;
			}
			uint rp = (h * push.r + r) * push.p + p;
			pre += dh * xv * load_f32(storage_buffers[push.mx_idx], rp);
			gsPre[r * MAXP + p] = pre;
			uint rankState = ((b * push.h + h) * push.r + r);
			store_f32(storage_buffers[push.v_state_idx], rankState * push.p + p,
				xv * load_f32(storage_buffers[push.mx_idx], rp));
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (push.has_norm != 0u && tid.x < push.r) {
		uint r = tid.x;
		float sumSq = 0.0f;
		for (uint j = 0u; j < push.p; ++j) {
			float value = gsPre[r * MAXP + j];
			sumSq += value * value;
		}
		gsInv[r] = rsqrt(sumSq / float(push.p) + push.eps);
	}
	GroupMemoryBarrierWithGroupSync();
	if (activeP) {
		uint xp = (b * push.h + h) * push.p + p;
		float z = load_f32(storage_buffers[push.z_idx], xp);
		float output = 0.0f;
		for (uint r = 0u; r < push.r; ++r) {
			uint rp = (h * push.r + r) * push.p + p;
			float core = gsPre[r * MAXP + p];
			if (push.has_norm != 0u)
				core *= gsInv[r] * load_f32(storage_buffers[push.norm_idx], h * push.p + p);
			float gate = push.has_z != 0u
				? silu(z * load_f32(storage_buffers[push.mz_idx], rp)) : 1.0f;
			output += core * gate * load_f32(storage_buffers[push.mo_idx], rp);
		}
		store_f32(storage_buffers[push.y_idx], xp, output);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint linear = tid.x; linear < push.r * push.n; linear += THREADS) {
		uint r = linear / push.n;
		uint n = linear % push.n;
		uint rankState = ((b * push.h + h) * push.r + r);
		store_f32(storage_buffers[push.k_state_idx], rankState * push.n + n, gsK[linear]);
	}
}
