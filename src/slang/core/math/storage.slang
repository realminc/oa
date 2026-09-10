// Canonical Core storage access for OA compute kernels.
//
// Element-to-byte addressing and storage conversion belong here rather than in
// individual handwritten or generated kernels. Future f16/bf16 helpers must
// preserve their packed-word concurrency, rounding, padding, and capability
// contracts from the proven OA implementations.

module storage;

[ForceInline]
public uint load_u8(RWByteAddressBuffer buffer, uint element_index) {
	uint aligned_byte_offset = element_index & ~3u;
	uint shift = (element_index & 3u) * 8u;
	return (buffer.Load(aligned_byte_offset) >> shift) & 0xffu;
}

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

[ForceInline]
public uint load_u32(RWByteAddressBuffer buffer, uint element_index) {
	return buffer.Load(element_index * 4u);
}

[ForceInline]
public void store_u32(RWByteAddressBuffer buffer, uint element_index, uint value) {
	buffer.Store(element_index * 4u, value);
}
