// Peak-preserving audio envelope for waveform display.
// Input:  [Channels, Samples] F32
// Output: [Bins, 2] F32 as {minimum, maximum}

import Storage;

struct PushConstants {
    uint a_idx;
    uint out_idx;
    uint channels;
    uint samples;
    uint bins;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint bin = tid.x;
    if (bin >= pc.bins) return;

    // Quotient/remainder partitioning avoids bin*samples overflow while
    // assigning every source sample to exactly one output bin.
    uint width = pc.samples / pc.bins;
    uint remainder = pc.samples % pc.bins;
    uint begin = bin * width + min(bin, remainder);
    uint end = begin + width + (bin < remainder ? 1 : 0);

    float seed = begin < end ? OaLoad(heap[pc.a_idx], begin) : 0.0;
    float minimum = seed;
    float maximum = seed;
    for (uint channel = 0; channel < pc.channels; ++channel) {
        uint base = channel * pc.samples;
        for (uint sample = begin; sample < end; ++sample) {
            float value = OaLoad(heap[pc.a_idx], base + sample);
            minimum = min(minimum, value);
            maximum = max(maximum, value);
        }
    }
    OaStore(heap[pc.out_idx], bin * 2, minimum);
    OaStore(heap[pc.out_idx], bin * 2 + 1, maximum);
}
