// Parameterized small-neighborhood NCHW filters.

import Image;
import Storage;

struct PushConstants {
    uint in_idx;
    uint out_idx;
    uint batch_size;
    uint channels;
    uint height;
    uint width;
    uint kernel_size;
    uint operation; // 0 median, 1 bilateral, 2 adaptive mean, 3 adaptive gaussian
    uint border_mode;
    float border_value;
    float p0;
    float p1;
    float p2;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint bc = gid.z;
    if (x >= pc.width || y >= pc.height || bc >= pc.batch_size * pc.channels) return;

    uint b = bc / pc.channels;
    uint c = bc % pc.channels;
    int radius = int(pc.kernel_size / 2);
    float center = OaLoadNchwBorder(heap[pc.in_idx], b, c, int(y), int(x),
        pc.channels, pc.height, pc.width, pc.border_mode, pc.border_value);
    float result = center;

    if (pc.operation == 0) {
        float samples[225];
        uint count = 0;
        for (uint ky = 0; ky < pc.kernel_size; ++ky) {
            for (uint kx = 0; kx < pc.kernel_size; ++kx) {
                float sample = OaLoadNchwBorder(heap[pc.in_idx], b, c,
                    int(y) + int(ky) - radius, int(x) + int(kx) - radius,
                    pc.channels, pc.height, pc.width, pc.border_mode, pc.border_value);
                uint position = count;
                while (position > 0 && samples[position - 1] > sample) {
                    samples[position] = samples[position - 1];
                    --position;
                }
                samples[position] = sample;
                ++count;
            }
        }
        result = samples[count / 2];
    } else {
        float weighted = 0.0f;
        float weight_sum = 0.0f;
        float sigma_space = max(pc.p0, 1.0e-6f);
        float sigma_color = max(pc.p1, 1.0e-6f);
        for (uint ky = 0; ky < pc.kernel_size; ++ky) {
            int dy = int(ky) - radius;
            for (uint kx = 0; kx < pc.kernel_size; ++kx) {
                int dx = int(kx) - radius;
                float sample = OaLoadNchwBorder(heap[pc.in_idx], b, c,
                    int(y) + dy, int(x) + dx, pc.channels, pc.height, pc.width,
                    pc.border_mode, pc.border_value);
                float weight = 1.0f;
                if (pc.operation == 1) {
                    float spatial = float(dx * dx + dy * dy);
                    float color = sample - center;
                    weight = exp(-spatial / (2.0f * sigma_space * sigma_space) -
                                 color * color / (2.0f * sigma_color * sigma_color));
                } else if (pc.operation == 3) {
                    float spatial = float(dx * dx + dy * dy);
                    weight = exp(-spatial / (2.0f * sigma_space * sigma_space));
                }
                weighted += sample * weight;
                weight_sum += weight;
            }
        }
        float filtered = weighted / max(weight_sum, 1.0e-20f);
        result = pc.operation == 1 ? filtered :
            (center > filtered - pc.p1 ? pc.p2 : 0.0f);
    }

    OaStoreNchw(heap[pc.out_idx], b, c, y, x,
        pc.channels, pc.height, pc.width, result);
}
