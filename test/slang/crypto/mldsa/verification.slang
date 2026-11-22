// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mldsa_verification;
import shake;

RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void verification_components(uint3 id : SV_DispatchThreadID) {
	uint stride = inputs.Load(0u);
	uint base = 4u + id.x * stride;
	uint pk_len = inputs.Load(base);
	uint sig_len = inputs.Load(base + 4u);
	uint ctx_len = inputs.Load(base + 8u);
	uint msg_len = inputs.Load(base + 12u);
	uint offset = inputs.Load(base + 16u);
	uint pk_base = base + 24u + offset;
	uint sig_base = pk_base + 1952u;
	uint ctx_base = sig_base + 3309u;
	uint msg_base = ctx_base + 255u;
	// Bound each record before invoking helpers, even if a corrupt length would
	// otherwise remain within the following record's physical buffer.
	uint reserved = 24u + offset + 1952u + 3309u + 255u;
	if (offset > 3u || reserved > stride || msg_len > stride - reserved) {
		outputs[id.x] = 0u;
		return;
	}
	outputs[id.x] = mldsa65_verify_external(inputs, pk_base, pk_len,
		inputs, sig_base, sig_len, inputs, msg_base, msg_len,
		inputs, ctx_base, ctx_len) ? 1u : 0u;
}

[numthreads(1, 1, 1)]
void range_components(uint3 id : SV_DispatchThreadID) {
	uint base = inputs.Load(id.x * 8u);
	uint length = inputs.Load(id.x * 8u + 4u);
	outputs[id.x] = pqc_buffer_range(inputs, base, length) ? 1u : 0u;
}

// Fixed laboratory record capacities encompass all FIPS parameter sets.
// Header carries set, logical lengths and byte alignment; stride bounds each row.
void parameter_verification_row(uint3 id, uint mode) {
	uint stride = inputs.Load(0u);
	uint base = 4u + id.x * stride;
	uint parameter_set = inputs.Load(base);
	uint pk_len = inputs.Load(base + 4u), sig_len = inputs.Load(base + 8u);
	uint ctx_len = inputs.Load(base + 12u), msg_len = inputs.Load(base + 16u);
	uint offset = inputs.Load(base + 20u);
	uint pk_base = base + 28u + offset;
	uint sig_base = pk_base + 2592u, ctx_base = sig_base + 4627u;
	uint msg_base = ctx_base + 255u;
	uint reserved = 28u + offset + 2592u + 4627u + 255u;
	if (offset > 3u || reserved > stride || msg_len > stride - reserved) {
		outputs[id.x] = 0u;
		return;
	}
	uint algorithm = inputs.Load(base + 24u);
	bool valid;
	if (mode == 2u) {
		valid = mldsa_verify_hash_message(parameter_set, inputs, pk_base, pk_len,
			inputs, sig_base, sig_len, inputs, msg_base, msg_len,
			inputs, ctx_base, ctx_len, algorithm);
	} else if (mode == 1u) {
		valid = mldsa_verify_prehashed(parameter_set, inputs, pk_base, pk_len,
			inputs, sig_base, sig_len, inputs, msg_base, msg_len,
			inputs, ctx_base, ctx_len, algorithm);
	} else {
		valid = mldsa_verify_external(parameter_set, inputs, pk_base, pk_len,
			inputs, sig_base, sig_len, inputs, msg_base, msg_len,
			inputs, ctx_base, ctx_len);
	}
	outputs[id.x] = valid ? 1u : 0u;
}

[numthreads(1, 1, 1)]
void parameter_verification_components(uint3 id : SV_DispatchThreadID) {
	parameter_verification_row(id, 0u);
}

[numthreads(1, 1, 1)]
void prehashed_verification_components(uint3 id : SV_DispatchThreadID) {
	parameter_verification_row(id, 1u);
}

[numthreads(1, 1, 1)]
void hash_message_verification_components(uint3 id : SV_DispatchThreadID) {
	parameter_verification_row(id, 2u);
}
