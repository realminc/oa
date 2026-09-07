// Canonical f32 storage access for OA compute kernels.
//
// Element-to-byte addressing and storage conversion belong here rather than in
// individual handwritten or generated kernels. Future f16/bf16 helpers must
// preserve their packed-word concurrency, rounding, padding, and capability
// contracts from the proven OA implementations.

module storage;

[ForceInline]
public float load_f32(RWByteAddressBuffer buffer, uint element_index) {
	return asfloat(buffer.Load(element_index * 4u));
}

[ForceInline]
public void store_f32(RWByteAddressBuffer buffer, uint element_index, float value) {
	buffer.Store(element_index * 4u, asuint(value));
}

[ForceInline]
public int load_i32(RWByteAddressBuffer buffer, uint element_index) {
	return asint(buffer.Load(element_index * 4u));
}

[ForceInline]
public void store_i32(RWByteAddressBuffer buffer, uint element_index, int value) {
	buffer.Store(element_index * 4u, asuint(value));
}
