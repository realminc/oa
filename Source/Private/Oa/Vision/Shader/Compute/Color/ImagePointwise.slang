// Parameterized NCHW pointwise image transforms.

import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint num_elements;
    uint operation;
    float p0;
    float p1;
    float p2;
    float p3;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.num_elements) return;

    float value = OaLoad(heap[pc.in_idx], i);
    float result = value;

    switch (pc.operation) {
        case 0: result = value > pc.p0 ? pc.p1 : 0.0f; break;                 // binary
        case 1: result = value > pc.p0 ? 0.0f : pc.p1; break;                 // binary inverse
        case 2: result = min(value, pc.p0); break;                            // truncate
        case 3: result = value > pc.p0 ? value : 0.0f; break;                 // to zero
        case 4: result = value > pc.p0 ? 0.0f : value; break;                 // to zero inverse
        case 5: result = value >= pc.p0 && value <= pc.p1 ? pc.p2 : 0.0f; break; // in range
        case 6: result = clamp(value, pc.p0, pc.p1); break;
        case 7: result = pc.p0 - value; break;                                // invert
        case 8: result = value * pc.p0 + pc.p1; break;                        // contrast, brightness
        case 9: result = pow(max(value, 0.0f), pc.p0) * pc.p1; break;         // gamma, gain
        case 10: result = value >= pc.p0 ? pc.p1 - value : value; break;      // solarize
        case 11: {                                                            // posterize
            float levels = max(pc.p0, 2.0f);
            float normalized = clamp((value - pc.p1) / (pc.p2 - pc.p1), 0.0f, 1.0f);
            // Explicit round-half-up keeps quantization stable across shader
            // compilers; HLSL round() is ties-to-even and makes 0.25 collapse
            // unexpectedly for a three-level [0,1] posterization.
            result = floor(normalized * (levels - 1.0f) + 0.5f) / (levels - 1.0f);
            result = result * (pc.p2 - pc.p1) + pc.p1;
            break;
        }
        default: break;
    }

    OaStore(heap[pc.out_idx], i, result);
}
