// SDK CartPole mix32 seed/lane/episode convention, applied to admitted episodes.
// Only completed lanes advance. Full reset reproduces the exact configured seed.
import storage;
import attributes;
import "common.slang";

struct PushConstants {
	uint market_index;
	uint features_index;
	uint bounds_index;
	uint ledger_index;
	uint meta_index;
	uint done_index;
	uint observation_index;
	uint environments;
	uint window;
	uint feature_width;
	uint horizon;
	uint seed_low;
	uint seed_high;
	uint only_done;
	float capital;
	uint market_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("trading_reset")]
[domain("ml")]
[variant("vectorized")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint lane = tid.x;
	if (lane >= push.environments) return;
	if (push.only_done != 0u && load_u8(storage_buffers[push.done_index],lane)==0u) return;
	uint episode = 0u;
	if (push.only_done != 0u) {
		episode = load_u32(storage_buffers[push.meta_index],lane*4u+2u);
		if (episode == 0xffffffffu || load_u32(storage_buffers[push.meta_index],lane*4u+3u)!=0u) {
			storage_buffers[push.meta_index].Store((lane*4u+3u)*4u,2u); return;
		}
		episode += 1u;
	}
	uint low = load_u32(storage_buffers[push.bounds_index],lane*2u);
	uint high = load_u32(storage_buffers[push.bounds_index],lane*2u+1u);
	uint hash = trading_mix32(push.seed_low ^ trading_mix32(push.seed_high+0x9e3779b9u));
	hash = trading_mix32(hash ^ trading_mix32(lane+0x85ebca6bu));
	hash = trading_mix32(hash ^ trading_mix32(episode+0xc2b2ae35u));
	// Modulo bias is documented; this is a reproducible episode schedule, not a
	// claim of uniform random sampling or a policy RNG substitute.
	uint row = low + hash % (high-low-push.horizon+1u);
	TradingLedger a;
	a.equity=push.capital; a.quantity=0.0f;
	a.mark=trading_row_mark(storage_buffers[push.market_index],row,push.market_width);
	a.peak=push.capital; a.fees=0.0f; a.funding=0.0f;
	a.slippage=0.0f; a.fills=0.0f; a.drawdown=0.0f; a.entry_price=0.0f; a.age=0.0f; a.position_peak=0.0f;
	a.pending_target=0.0f;
	a.pending_delay=0.0f;
	a.pending_limit=0.0f;
	a.pending_size=0.0f;
	a.liquidations=0.0f;
	a.unfilled_iocs=0.0f;
	a.initial_margin=0.0f;
	a.maintenance_margin=0.0f;

	trading_store(storage_buffers[push.ledger_index],lane,a);
	store_u32(storage_buffers[push.meta_index],lane*4u,row);
	store_u32(storage_buffers[push.meta_index],lane*4u+1u,0u);
	store_u32(storage_buffers[push.meta_index],lane*4u+2u,episode);
	store_u32(storage_buffers[push.meta_index],lane*4u+3u,0u);
	store_u8(storage_buffers[push.done_index],lane,0u);
	trading_observe(storage_buffers[push.observation_index],storage_buffers[push.features_index],lane,row,push.window,push.feature_width,a,push.capital,0u,push.horizon);
}
