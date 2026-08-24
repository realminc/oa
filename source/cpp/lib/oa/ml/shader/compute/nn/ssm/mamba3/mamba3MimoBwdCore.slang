// Reverse scan for the shared MIMO state. Q/K remain in rotated coordinates;
// Mamba3MimoBwdRotate owns the inverse rotary transform and angle adjoint.

import storage;
import activations;

static const uint MAXPN = 4096u;

struct PushConstants {
	uint dpre_idx, x_idx, d_idx, mx_idx, q_idx, k_idx, hprev_idx;
	uint decay_idx, gamma_idx, scale_idx, dt_idx, trap_idx;
	uint dq_idx, dk_idx, dx_idx, dadt_idx, ddt_idx, dtrap_idx;
	uint dd_batch_idx, dmx_batch_idx, dscale_tmp_idx;
	uint B, L, H, P, N, R, has_d;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	float dState[MAXPN];
	float u[MAXPN];
	float dU[MAXPN];
	for (uint i = 0u; i < pc.P * pc.N; ++i) dState[i] = 0.0f;
	uint mimoBatch = bh * pc.R * pc.P;
	for (uint i = 0u; i < pc.R * pc.P; ++i)
		oaStoreF32(heap[pc.dmx_batch_idx], mimoBatch + i, 0.0f);
	oaStoreF32(heap[pc.dd_batch_idx], bh, 0.0f);
	float dh = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;

	for (int ti = int(pc.L) - 1; ti >= 0; --ti) {
		uint t = uint(ti);
		uint scalar = (b * pc.L + t) * pc.H + h;
		uint xBase = scalar * pc.P;
		float decay = oaLoadF32(heap[pc.decay_idx], scalar);
		float gamma = oaLoadF32(heap[pc.gamma_idx], scalar);
		float scale = oaLoadF32(heap[pc.scale_idx], scalar);
		for (uint p = 0u; p < pc.P; ++p) {
			float xv = oaLoadF32(heap[pc.x_idx], xBase + p);
			for (uint n = 0u; n < pc.N; ++n) {
				float value = 0.0f;
				for (uint r = 0u; r < pc.R; ++r) {
					uint rp = (h * pc.R + r) * pc.P + p;
					uint qk = (scalar * pc.R + r) * pc.N + n;
					value += xv * oaLoadF32(heap[pc.mx_idx], rp)
						* oaLoadF32(heap[pc.k_idx], qk);
				}
				u[p * pc.N + n] = value;
			}
		}

		float dScale = 0.0f, dDecay = 0.0f, dGamma = 0.0f;
		for (uint p = 0u; p < pc.P; ++p) {
			uint history = ((bh * pc.L + t) * pc.P + p) * pc.N;
			for (uint n = 0u; n < pc.N; ++n) {
				float queryGrad = 0.0f;
				for (uint r = 0u; r < pc.R; ++r) {
					uint qk = (scalar * pc.R + r) * pc.N + n;
					queryGrad += oaLoadF32(heap[pc.dpre_idx],
						(scalar * pc.R + r) * pc.P + p)
						* oaLoadF32(heap[pc.q_idx], qk);
				}
				uint stateIndex = p * pc.N + n;
				float incoming = dState[stateIndex];
				float dHd = incoming + queryGrad;
				dScale += incoming * u[stateIndex];
				dGamma += queryGrad * u[stateIndex];
				dDecay += dHd * oaLoadF32(heap[pc.hprev_idx], history + n);
				dU[stateIndex] = scale * incoming + gamma * queryGrad;
				dState[stateIndex] = decay * dHd;
			}
		}
		dGamma += dScale;
		oaStoreF32(heap[pc.dscale_tmp_idx], scalar, dScale);
		oaStoreF32(heap[pc.ddt_idx], scalar, dGamma);
		oaStoreF32(heap[pc.dadt_idx], scalar, dDecay * decay);

		for (uint r = 0u; r < pc.R; ++r) {
			for (uint n = 0u; n < pc.N; ++n) {
				float dQ = 0.0f, dK = 0.0f;
				for (uint p = 0u; p < pc.P; ++p) {
					uint history = ((bh * pc.L + t) * pc.P + p) * pc.N;
					float state = decay * oaLoadF32(heap[pc.hprev_idx], history + n)
						+ gamma * u[p * pc.N + n];
					dQ += oaLoadF32(heap[pc.dpre_idx],
						(scalar * pc.R + r) * pc.P + p) * state;
					float v = oaLoadF32(heap[pc.x_idx], xBase + p)
						* oaLoadF32(heap[pc.mx_idx], (h * pc.R + r) * pc.P + p);
					dK += dU[p * pc.N + n] * v;
				}
				uint qk = (scalar * pc.R + r) * pc.N + n;
				oaStoreF32(heap[pc.dq_idx], qk, dQ);
				oaStoreF32(heap[pc.dk_idx], qk, dK);
			}
		}

		float dD = 0.0f;
		for (uint p = 0u; p < pc.P; ++p) {
			float dx = 0.0f;
			float xv = oaLoadF32(heap[pc.x_idx], xBase + p);
			for (uint r = 0u; r < pc.R; ++r) {
				float dV = pc.has_d != 0u ? dh * oaLoadF32(heap[pc.dpre_idx],
					(scalar * pc.R + r) * pc.P + p) : 0.0f;
				for (uint n = 0u; n < pc.N; ++n) {
					uint qk = (scalar * pc.R + r) * pc.N + n;
					dV += dU[p * pc.N + n] * oaLoadF32(heap[pc.k_idx], qk);
				}
				uint rp = (h * pc.R + r) * pc.P + p;
				dx += dV * oaLoadF32(heap[pc.mx_idx], rp);
				uint partial = mimoBatch + r * pc.P + p;
				oaStoreF32(heap[pc.dmx_batch_idx], partial,
					oaLoadF32(heap[pc.dmx_batch_idx], partial) + dV * xv);
				dD += oaLoadF32(heap[pc.dpre_idx],
					(scalar * pc.R + r) * pc.P + p)
					* xv * oaLoadF32(heap[pc.mx_idx], rp);
			}
			oaStoreF32(heap[pc.dx_idx], xBase + p, dx);
		}
		oaStoreF32(heap[pc.dd_batch_idx], bh,
			oaLoadF32(heap[pc.dd_batch_idx], bh) + dD);
	}

	for (uint t = 0u; t < pc.L; ++t) {
		uint scalar = (b * pc.L + t) * pc.H + h;
		float sigmoidTrap = Sigmoid(oaLoadF32(heap[pc.trap_idx], scalar));
		float dGamma = oaLoadF32(heap[pc.ddt_idx], scalar);
		float dShift = 0.0f;
		if (t > 0u) {
			uint previous = (b * pc.L + t - 1u) * pc.H + h;
			dShift = oaLoadF32(heap[pc.dscale_tmp_idx], previous);
		}
		float dt = oaLoadF32(heap[pc.dt_idx], scalar);
		oaStoreF32(heap[pc.ddt_idx], scalar,
			dGamma * sigmoidTrap + dShift * (1.0f - sigmoidTrap));
		oaStoreF32(heap[pc.dtrap_idx], scalar,
			dt * (dGamma - dShift) * sigmoidTrap * (1.0f - sigmoidTrap));
	}
}
