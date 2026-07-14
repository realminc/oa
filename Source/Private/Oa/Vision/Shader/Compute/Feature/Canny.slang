// Canny.slang — Canny Edge Detection
// pass=0: NMS, pass=1: double threshold, pass=2: hysteresis

import Storage;

struct PushConstants {
    uint mag_idx;
    uint dir_idx;
    uint out_idx;
    uint batch_size;
    uint height;
    uint width;
    float low_thresh;
    float high_thresh;
    uint pass;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float LoadNchw(uint buf_idx, uint b, int h, int w) {
    if (h < 0 || h >= int(pc.height) || w < 0 || w >= int(pc.width)) return 0.0f;
    uint idx = b * (pc.height * pc.width) + uint(h) * pc.width + uint(w);
    return OaLoad(heap[buf_idx], idx);
}

void StoreNchw(uint buf_idx, uint b, uint h, uint w, float v) {
    uint idx = b * (pc.height * pc.width) + h * pc.width + w;
    OaStore(heap[buf_idx], idx, v);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint x = gid.x * 16 + lid.x;
    uint y = gid.y * 16 + lid.y;
    uint n = gid.z;
    if (x >= pc.width || y >= pc.height || n >= pc.batch_size) { return; }

    // Pass 0: Non-maximum suppression
    if (pc.pass == 0) {
        float mag = LoadNchw(pc.mag_idx, n, int(y), int(x));
        float dir = LoadNchw(pc.dir_idx, n, int(y), int(x));

        float a = dir;
        if (a < 0.0f) { a += 3.14159265f; }
        float deg = a * 57.2957795f;

        float n1 = 0.0f;
        float n2 = 0.0f;
        if (deg < 22.5f || deg >= 157.5f) {
            n1 = LoadNchw(pc.mag_idx, n, int(y), int(x - 1));
            n2 = LoadNchw(pc.mag_idx, n, int(y), int(x + 1));
        } else if (deg < 67.5f) {
            n1 = LoadNchw(pc.mag_idx, n, int(y + 1), int(x - 1));
            n2 = LoadNchw(pc.mag_idx, n, int(y - 1), int(x + 1));
        } else if (deg < 112.5f) {
            n1 = LoadNchw(pc.mag_idx, n, int(y - 1), int(x));
            n2 = LoadNchw(pc.mag_idx, n, int(y + 1), int(x));
        } else {
            n1 = LoadNchw(pc.mag_idx, n, int(y - 1), int(x - 1));
            n2 = LoadNchw(pc.mag_idx, n, int(y + 1), int(x + 1));
        }
        float r = (mag >= n1 && mag >= n2) ? mag : 0.0f;
        StoreNchw(pc.mag_idx, n, y, x, r);
        return;
    }

    // Pass 1: Double threshold
    if (pc.pass == 1) {
        float mag = LoadNchw(pc.mag_idx, n, int(y), int(x));
        float ev = 0.0f;
        if (mag >= pc.high_thresh) {
            ev = 255.0f;
        } else if (mag >= pc.low_thresh) {
            ev = 128.0f;
        }
        StoreNchw(pc.out_idx, n, y, x, ev);
        return;
    }

    // Pass 2: Edge tracking by hysteresis
    if (pc.pass == 2) {
        float edge = LoadNchw(pc.out_idx, n, int(y), int(x));
        if (edge < 127.0f || edge > 129.0f) { return; }
        bool hs = false;
        int dy = -1;
        while (dy <= 1 && !hs) {
            int dx = -1;
            while (dx <= 1 && !hs) {
                if (!(dy == 0 && dx == 0)) {
                    int ny = int(y) + dy;
                    int nx = int(x) + dx;
                    if (!(ny < 0 || ny >= int(pc.height) || nx < 0 || nx >= int(pc.width))) {
                        float ng = LoadNchw(pc.out_idx, n, ny, nx);
                        if (ng >= 254.0f) { hs = true; }
                    }
                }
                dx = dx + 1;
            }
            dy = dy + 1;
        }
        if (hs) {
            StoreNchw(pc.out_idx, n, y, x, 255.0f);
        } else {
            StoreNchw(pc.out_idx, n, y, x, 0.0f);
        }
        return;
    }
}