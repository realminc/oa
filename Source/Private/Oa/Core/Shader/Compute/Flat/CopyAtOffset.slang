// Copy src[0..count-1] into dst[dst_offset..dst_offset+count-1].
// Used by Slice DescribeBackward to scatter gradient into the correct slice position.

import Storage;

struct PushConstants {
    uint src_idx;     // prepended by Graph.cpp/Stream.cpp
    uint dst_idx;     // prepended by Graph.cpp/Stream.cpp
    uint count;
    uint dst_offset;  // element offset into destination buffer
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.count) return;
    OaStore(heap[pc.dst_idx], pc.dst_offset + idx, OaLoad(heap[pc.src_idx], idx));
}
