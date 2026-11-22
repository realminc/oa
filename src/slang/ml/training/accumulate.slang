// OA donor: core flat add.gen.slang. Explicit reset avoids reading old state.
import storage;
import attributes;
struct PushConstants {
 uint target_index; uint value_index; uint cursor_index; uint count; float scale;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("training_accumulate")]
[domain("ml")]
[variant("cursor_reset")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
 uint i = tid.x;
 if (i >= push.count) return;
 float value = push.scale * load_f32(storage_buffers[push.value_index], i);
 if (load_u32(storage_buffers[push.cursor_index], 1) == 0)
  value += load_f32(storage_buffers[push.target_index], i);
 store_f32(storage_buffers[push.target_index], i, value);
}
