// StorageBf16.slang — Pure BF16 storage (2 bytes/element)
// No DTYPE branching. Always uses bfloat16 layout.
// BF16 = high 16 bits of FP32 (sign + 8-bit exp + 7-bit mantissa)

// Convert BF16 (uint16) to FP32
float Bf16ToF32(uint bf16_bits) {
    return asfloat(bf16_bits << 16);
}

// Convert FP32 to BF16 (uint16) with round-to-nearest-even
uint F32ToBf16(float f32_val) {
    uint bits = asuint(f32_val);
    uint rounding = ((bits >> 16) & 1) + 0x7FFF;
    return (bits + rounding) >> 16;
}

// Load BF16 from buffer at element index, return as FP32
float OaLoad(RWByteAddressBuffer buf, uint idx) {
    uint byte_offset = idx * 2;
    uint word_offset = byte_offset & ~3;
    uint shift = (byte_offset & 2) * 8;
    uint word = buf.Load(word_offset);
    uint bf16_bits = (word >> shift) & 0xFFFF;
    return Bf16ToF32(bf16_bits);
}

// Store FP32 to buffer as BF16 at element index
void OaStore(RWByteAddressBuffer buf, uint idx, float value) {
    uint bf16_bits = F32ToBf16(value);
    uint byte_offset = idx * 2;
    uint word_offset = byte_offset & ~3;
    uint shift = (byte_offset & 2) * 8;
    
    uint mask = 0xFFFF << shift;
    uint new_bits = bf16_bits << shift;
    
    uint old_word = 0;
    uint assumed;
    do {
        assumed = old_word;
        uint new_word = (assumed & ~mask) | new_bits;
        buf.InterlockedCompareExchange(word_offset, assumed, new_word, old_word);
    } while (assumed != old_word);
}

// Atomic add for BF16 (read-modify-write in FP32, store as BF16)
void OaAtomicAdd(RWByteAddressBuffer buf, uint idx, float value) {
    uint byte_offset = idx * 2;
    uint word_offset = byte_offset & ~3;
    uint shift = (byte_offset & 2) * 8;
    uint mask = 0xFFFF << shift;
    
    uint old_word = 0;
    uint assumed;
    do {
        assumed = old_word;
        uint old_bf16 = (assumed >> shift) & 0xFFFF;
        float old_f32 = Bf16ToF32(old_bf16);
        float new_f32 = old_f32 + value;
        uint new_bf16 = F32ToBf16(new_f32);
        uint new_bits = new_bf16 << shift;
        uint new_word = (assumed & ~mask) | new_bits;
        
        buf.InterlockedCompareExchange(word_offset, assumed, new_word, old_word);
    } while (assumed != old_word);
}
