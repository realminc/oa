// Mechanical graph-resident advancement of the existing host-written PPO
// cursor. Philox seeds retain exact low/high 32-bit carry arithmetic.
import storage;
import attributes;
struct PushConstants { uint cursor_index; uint horizon; uint continuing; };
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("trading_cursor_advance")]
[domain("ml")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(1,1,1)]
void main() {
	uint step=load_u32(storage_buffers[push.cursor_index],0)+1u;
	store_u32(storage_buffers[push.cursor_index],0,step);
	if (step<push.horizon) {
		uint low=load_u32(storage_buffers[push.cursor_index],1)+1u;
		store_u32(storage_buffers[push.cursor_index],1,low);
		if (low==0u) store_u32(storage_buffers[push.cursor_index],2,load_u32(storage_buffers[push.cursor_index],2)+1u);
	}
	store_u32(storage_buffers[push.cursor_index],5,(push.continuing==0u && step+1u==push.horizon) ? 1u : 0u);
}
