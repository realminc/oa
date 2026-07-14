// StorageFp32.slang — Pure FP32 storage (4 bytes/element)
// No DTYPE branching. Always uses float32 layout.

// Load float32 from buffer at element index
float OaLoad(RWByteAddressBuffer buf, uint idx) {
    return asfloat(buf.Load(idx * 4));
}

// Store float32 to buffer at element index
void OaStore(RWByteAddressBuffer buf, uint idx, float value) {
    buf.Store(idx * 4, asuint(value));
}

// Atomic add for float32
void OaAtomicAdd(RWByteAddressBuffer buf, uint idx, float value) {
    uint addr = idx * 4;
    uint old_val = 0;
    uint assumed;
    do {
        assumed = old_val;
        uint new_val = asuint(asfloat(assumed) + value);
        buf.InterlockedCompareExchange(addr, assumed, new_val, old_val);
    } while (assumed != old_val);
}