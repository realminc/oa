// Reverse scan for the shared MIMO state. Q/K remain in rotated coordinates;
// Mamba3MimoBwdRotate owns the inverse rotary transform and angle adjoint.

import storage;
import activations;
import attributes;

static const uint MAXPN = 4096u;

struct PushConstants {
	uint dpre_idx, x_idx, d_idx, mx_idx, q_idx, k_idx, hprev_idx;
	uint decay_idx, gamma_idx, scale_idx, dt_idx, trap_idx;
	uint dq_idx, dk_idx, dx_idx, dadt_idx, ddt_idx, dtrap_idx;
	uint dd_batch_idx, dmx_batch_idx, dscale_tmp_idx;
	uint b, l, h, p, n, r, has_d;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_mimo_backward_core")]
[domain("ml")]
[variant("reverse")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	float dState[MAXPN];
	float u[MAXPN];
	float dU[MAXPN];
	for (uint i = 0u; i < push.p * push.n; ++i) dState[i] = 0.0f;
	uint mimoBatch = bh * push.r * push.p;
	for (uint i = 0u; i < push.r * push.p; ++i)
		store_f32(storage_buffers[push.dmx_batch_idx], mimoBatch + i, 0.0f);
	store_f32(storage_buffers[push.dd_batch_idx], bh, 0.0f);
	float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;

	for (int ti = int(push.l) - 1; ti >= 0; --ti) {
		uint t = uint(ti);
		uint scalar = (b * push.l + t) * push.h + h;
		uint xBase = scalar * push.p;
		float decay = load_f32(storage_buffers[push.decay_idx], scalar);
		float gamma = load_f32(storage_buffers[push.gamma_idx], scalar);
		float scale = load_f32(storage_buffers[push.scale_idx], scalar);
		for (uint p = 0u; p < push.p; ++p) {
			float xv = load_f32(storage_buffers[push.x_idx], xBase + p);
			for (uint n = 0u; n < push.n; ++n) {
				float value = 0.0f;
				for (uint r = 0u; r < push.r; ++r) {
					uint rp = (h * push.r + r) * push.p + p;
					uint qk = (scalar * push.r + r) * push.n + n;
					value += xv * load_f32(storage_buffers[push.mx_idx], rp)
						* load_f32(storage_buffers[push.k_idx], qk);
				}
				u[p * push.n + n] = value;
			}
		}

		float dScale = 0.0f, dDecay = 0.0f, dGamma = 0.0f;
		for (uint p = 0u; p < push.p; ++p) {
			uint history = ((bh * push.l + t) * push.p + p) * push.n;
			for (uint n = 0u; n < push.n; ++n) {
				float queryGrad = 0.0f;
				for (uint r = 0u; r < push.r; ++r) {
					uint qk = (scalar * push.r + r) * push.n + n;
					queryGrad += load_f32(storage_buffers[push.dpre_idx],
						(scalar * push.r + r) * push.p + p)
						* load_f32(storage_buffers[push.q_idx], qk);
				}
				uint stateIndex = p * push.n + n;
				float incoming = dState[stateIndex];
				float dHd = incoming + queryGrad;
				dScale += incoming * u[stateIndex];
				dGamma += queryGrad * u[stateIndex];
				dDecay += dHd * load_f32(storage_buffers[push.hprev_idx], history + n);
				dU[stateIndex] = scale * incoming + gamma * queryGrad;
				dState[stateIndex] = decay * dHd;
			}
		}
		dGamma += dScale;
		store_f32(storage_buffers[push.dscale_tmp_idx], scalar, dScale);
		store_f32(storage_buffers[push.ddt_idx], scalar, dGamma);
		store_f32(storage_buffers[push.dadt_idx], scalar, dDecay * decay);

		for (uint r = 0u; r < push.r; ++r) {
			for (uint n = 0u; n < push.n; ++n) {
				float dQ = 0.0f, dK = 0.0f;
				for (uint p = 0u; p < push.p; ++p) {
					uint history = ((bh * push.l + t) * push.p + p) * push.n;
					float state = decay * load_f32(storage_buffers[push.hprev_idx], history + n)
						+ gamma * u[p * push.n + n];
					dQ += load_f32(storage_buffers[push.dpre_idx],
						(scalar * push.r + r) * push.p + p) * state;
					float v = load_f32(storage_buffers[push.x_idx], xBase + p)
						* load_f32(storage_buffers[push.mx_idx], (h * push.r + r) * push.p + p);
					dK += dU[p * push.n + n] * v;
				}
				uint qk = (scalar * push.r + r) * push.n + n;
				store_f32(storage_buffers[push.dq_idx], qk, dQ);
				store_f32(storage_buffers[push.dk_idx], qk, dK);
			}
		}

		float dD = 0.0f;
		for (uint p = 0u; p < push.p; ++p) {
			float dx = 0.0f;
			float xv = load_f32(storage_buffers[push.x_idx], xBase + p);
			for (uint r = 0u; r < push.r; ++r) {
				float dV = push.has_d != 0u ? dh * load_f32(storage_buffers[push.dpre_idx],
					(scalar * push.r + r) * push.p + p) : 0.0f;
				for (uint n = 0u; n < push.n; ++n) {
					uint qk = (scalar * push.r + r) * push.n + n;
					dV += dU[p * push.n + n] * load_f32(storage_buffers[push.k_idx], qk);
				}
				uint rp = (h * push.r + r) * push.p + p;
				dx += dV * load_f32(storage_buffers[push.mx_idx], rp);
				uint partial = mimoBatch + r * push.p + p;
				store_f32(storage_buffers[push.dmx_batch_idx], partial,
					load_f32(storage_buffers[push.dmx_batch_idx], partial) + dV * xv);
				dD += load_f32(storage_buffers[push.dpre_idx],
					(scalar * push.r + r) * push.p + p)
					* xv * load_f32(storage_buffers[push.mx_idx], rp);
			}
			store_f32(storage_buffers[push.dx_idx], xBase + p, dx);
		}
		store_f32(storage_buffers[push.dd_batch_idx], bh,
			load_f32(storage_buffers[push.dd_batch_idx], bh) + dD);
	}

	for (uint t = 0u; t < push.l; ++t) {
		uint scalar = (b * push.l + t) * push.h + h;
		float sigmoidTrap = sigmoid(load_f32(storage_buffers[push.trap_idx], scalar));
		float dGamma = load_f32(storage_buffers[push.ddt_idx], scalar);
		float dShift = 0.0f;
		if (t > 0u) {
			uint previous = (b * push.l + t - 1u) * push.h + h;
			dShift = load_f32(storage_buffers[push.dscale_tmp_idx], previous);
		}
		float dt = load_f32(storage_buffers[push.dt_idx], scalar);
		store_f32(storage_buffers[push.ddt_idx], scalar,
			dGamma * sigmoidTrap + dShift * (1.0f - sigmoidTrap));
		store_f32(storage_buffers[push.dtrap_idx], scalar,
			dt * (dGamma - dShift) * sigmoidTrap * (1.0f - sigmoidTrap));
	}
}
