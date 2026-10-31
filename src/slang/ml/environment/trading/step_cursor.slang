// Deliberate economic correction of tradecpp trainer.cu, not CUDA profit parity.
// One invocation owns one lane. Funding coefficient is native rate*oracle for
// the next open, supplied by the consumer; no estimation from candle returns.
import storage;
import attributes;
import "common.slang";

struct PushConstants {
	uint market_index;
	uint features_index;
	uint bounds_index;
	uint action_index;
	uint ledger_index;
	uint meta_index;
	uint done_index;
	uint observation_index;
	uint reward_index;
	uint terminated_index;
	uint truncated_index;
	uint cursor_index;
	uint rules_index;
	uint environments;
	uint window;
	uint feature_width;
	uint horizon;
	uint perpetual;
	uint liquidate_at_horizon;
	float capital;
	float exposure;
	float taker_fee;
	float slippage;
	float max_drawdown;
	uint market_width;
	float book_slippage;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("trading_step_cursor")]
[domain("ml")]
[variant("next_open_cursor")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint lane=tid.x;
	if (lane>=push.environments) return;
	uint row=load_u32(storage_buffers[push.meta_index],lane*4u);
	uint steps=load_u32(storage_buffers[push.meta_index],lane*4u+1u);
	uint high=load_u32(storage_buffers[push.bounds_index],lane*2u+1u);
	int action=load_i32(storage_buffers[push.action_index],lane);
	bool invalid=action<0 || action>(push.perpetual!=0u ? 2 : 1)
		|| row>=high || steps>=push.horizon || load_u8(storage_buffers[push.done_index],lane)!=0u
		|| load_u32(storage_buffers[push.meta_index],lane*4u+3u)!=0u;
	TradingLedger a=trading_load(storage_buffers[push.ledger_index],lane);
	TradingLedger before=a;
	bool terminal=false;
	bool truncated=false;
	if (!invalid) {
		uint next=row+1u;
		bool boundary=steps+1u>=push.horizon || load_u32(storage_buffers[push.cursor_index],5)!=0u;
		int target=push.perpetual!=0u ? action-1 : action;
		invalid=!trading_advance(a,target,storage_buffers[push.market_index],row,next,push.market_width,
			storage_buffers[push.rules_index],lane,push.exposure,push.taker_fee,push.slippage,
			push.book_slippage,push.max_drawdown,boundary,push.liquidate_at_horizon!=0u,terminal);
		steps+=1u;
		truncated=boundary && !terminal;
		invalid=invalid || !isfinite(a.equity/push.capital) || !isfinite(a.peak/push.capital)
			|| !isfinite(a.quantity*a.mark/push.capital) || !isfinite(a.position_peak/push.capital)
			|| !isfinite(a.initial_margin/push.capital) || !isfinite(a.maintenance_margin/push.capital);
		if (!invalid) {
			row=next;
			trading_store(storage_buffers[push.ledger_index],lane,a);
			store_u32(storage_buffers[push.meta_index],lane*4u,row);
			store_u32(storage_buffers[push.meta_index],lane*4u+1u,steps);
		}
	}
	if (invalid) {
		a=before; terminal=true; truncated=false;
		store_u32(storage_buffers[push.meta_index],lane*4u+3u,1u);
	}
	store_f32(storage_buffers[push.reward_index],lane,invalid ? 0.0f : (a.equity-before.equity)/push.capital);
	store_u8(storage_buffers[push.terminated_index],lane,terminal ? 1u : 0u);
	store_u8(storage_buffers[push.truncated_index],lane,truncated ? 1u : 0u);
	store_u8(storage_buffers[push.done_index],lane,terminal || truncated ? 1u : 0u);
	trading_observe(storage_buffers[push.observation_index],storage_buffers[push.features_index],lane,row,push.window,push.feature_width,a,push.capital,steps,push.horizon);
}
