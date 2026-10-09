// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mldsa_packing;

RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void packing_components(uint3 tid : SV_DispatchThreadID) {
	// One padded record: offset header, at most three leading poison bytes,
	// 1952-byte public key, 3309-byte signature, trailing poison padding.
	uint base = tid.x * 5268u;
	uint pk = base + 4u + inputs.Load(base);
	uint sig = pk + 1952u;
	uint rho[8], t1[6][256], challenge[12], z[5][256], h[6][256];
	mldsa65_unpack_pk(inputs, pk, rho, t1);
	mldsa65_unpack_ctilde(inputs, sig, challenge);
	mldsa65_unpack_z(inputs, sig, z);
	bool valid = mldsa65_unpack_h(inputs, sig, h);
	bool norm = true;
	uint o = tid.x * 4374u;
	for (uint i = 0u; i < 8u; ++i) outputs[o++] = rho[i];
	for (uint k = 0u; k < 6u; ++k)
		for (uint i = 0u; i < 256u; ++i) outputs[o++] = t1[k][i];
	for (uint i = 0u; i < 12u; ++i) outputs[o++] = challenge[i];
	for (uint l = 0u; l < 5u; ++l) {
		norm = norm && mldsa_check_norm(z[l], 524092u);
		for (uint i = 0u; i < 256u; ++i) outputs[o++] = z[l][i];
	}
	for (uint k = 0u; k < 6u; ++k)
		for (uint i = 0u; i < 256u; ++i) outputs[o++] = h[k][i];
	outputs[o++] = valid ? 1u : 0u;
	outputs[o] = norm ? 1u : 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bit_packing_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 652u;
	uint bits = inputs.Load(base);
	uint offset = inputs.Load(base + 4u);
	for (uint i = 0u; i < 256u; ++i) {
		outputs[tid.x * 256u + i] =
			unpack_unsigned_coeff(inputs, base + 8u + offset, i, bits);
	}
}
