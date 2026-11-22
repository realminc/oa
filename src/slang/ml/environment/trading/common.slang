// Trading research contract v1. Corrects tradecpp trainer.cu's close-to-close
// action timing, percentage exposure rebalancing, one-cost reversals and reward
// mislabeled as raw profit. Fixed signed quantity and next-open mark/fills match
// Trade's independent Decimal oracle. FP32 simulation never supplies live money.
module trading_common;
import storage;

public uint trading_mix32(uint x) {
	x ^= x >> 16u; x *= 0x7feb352du;
	x ^= x >> 15u; x *= 0x846ca68bu;
	x ^= x >> 16u; return x;
}

public struct TradingLedger {
	public float equity;
	public float quantity;
	public float mark;
	public float peak;
	public float fees;
	public float funding;
	public float slippage;
	public float fills;
	public float drawdown;
	public float entry_price;
	public float age;
	public float position_peak;
	public float pending_target;
	public float pending_delay;
	public float pending_limit;
	public float pending_size;
	public float liquidations;
	public float unfilled_iocs;
	public float initial_margin;
	public float maintenance_margin;
	public float maker_pending;
	public float maker_price;
	public float maker_deadline;
	public float maker_requotes;
	public float maker_fills;
	public float maker_fallbacks;

};

public TradingLedger trading_load(RWByteAddressBuffer buffer, uint lane) {
	uint b = lane * 26u;
	TradingLedger a;
	a.equity = load_f32(buffer,b); a.quantity = load_f32(buffer,b+1u);
	a.mark = load_f32(buffer,b+2u); a.peak = load_f32(buffer,b+3u);
	a.fees = load_f32(buffer,b+4u); a.funding = load_f32(buffer,b+5u);
	a.slippage = load_f32(buffer,b+6u); a.fills = load_f32(buffer,b+7u);
	a.drawdown = load_f32(buffer,b+8u); a.entry_price = load_f32(buffer,b+9u); a.age = load_f32(buffer,b+10u); a.position_peak = load_f32(buffer,b+11u);
	a.pending_target = load_f32(buffer,b+12u);
	a.pending_delay = load_f32(buffer,b+13u);
	a.pending_limit = load_f32(buffer,b+14u);
	a.pending_size = load_f32(buffer,b+15u);
	a.liquidations = load_f32(buffer,b+16u);
	a.unfilled_iocs = load_f32(buffer,b+17u);
	a.initial_margin = load_f32(buffer,b+18u);
	a.maintenance_margin = load_f32(buffer,b+19u);
	a.maker_pending = load_f32(buffer,b+20u);
	a.maker_price = load_f32(buffer,b+21u);
	a.maker_deadline = load_f32(buffer,b+22u);
	a.maker_requotes = load_f32(buffer,b+23u);
	a.maker_fills = load_f32(buffer,b+24u);
	a.maker_fallbacks = load_f32(buffer,b+25u);
	return a;
}

public void trading_store(RWByteAddressBuffer buffer, uint lane, TradingLedger a) {
	uint b = lane * 26u;
	store_f32(buffer,b,a.equity); store_f32(buffer,b+1u,a.quantity);
	store_f32(buffer,b+2u,a.mark); store_f32(buffer,b+3u,a.peak);
	store_f32(buffer,b+4u,a.fees); store_f32(buffer,b+5u,a.funding);
	store_f32(buffer,b+6u,a.slippage); store_f32(buffer,b+7u,a.fills);
	store_f32(buffer,b+8u,a.drawdown);
	store_f32(buffer,b+9u,a.entry_price); store_f32(buffer,b+10u,a.age); store_f32(buffer,b+11u,a.position_peak);
	store_f32(buffer,b+12u,a.pending_target);
	store_f32(buffer,b+13u,a.pending_delay);
	store_f32(buffer,b+14u,a.pending_limit);
	store_f32(buffer,b+15u,a.pending_size);
	store_f32(buffer,b+16u,a.liquidations);
	store_f32(buffer,b+17u,a.unfilled_iocs);
	store_f32(buffer,b+18u,a.initial_margin);
	store_f32(buffer,b+19u,a.maintenance_margin);
	store_f32(buffer,b+20u,a.maker_pending);
	store_f32(buffer,b+21u,a.maker_price);
	store_f32(buffer,b+22u,a.maker_deadline);
	store_f32(buffer,b+23u,a.maker_requotes);
	store_f32(buffer,b+24u,a.maker_fills);
	store_f32(buffer,b+25u,a.maker_fallbacks);

}

public void trading_mark(inout TradingLedger a, float price) {
	a.equity += a.quantity * (price - a.mark); a.mark = price;
	// Only observed marks, never candle high/low or future extrema.
	if (a.quantity != 0.0f) a.position_peak = max(a.position_peak,a.quantity * (price - a.entry_price));
}

public bool trading_risk(inout TradingLedger a, float limit) {
	a.peak = max(a.peak,a.equity);
	float dd = (a.peak - a.equity) / a.peak;
	a.drawdown = max(a.drawdown,dd);
	return a.equity <= 0.0f || dd >= limit;
}

public float trading_fill(
	inout TradingLedger a,
	float delta,
	float price,
	float fee,
	float slip,
	float fee_per_unit
) {
	float fill = price * (1.0f + (delta > 0.0f ? slip : -slip));
	float cost = abs(delta) * (fill * fee + fee_per_unit);
	float drag = delta * (fill - price);
	a.equity -= cost + drag; a.fees += cost; a.slippage += drag; a.fills += 1.0f;
	return fill;
}

public void trading_rebalance(
	inout TradingLedger a,
	int target,
	float price,
	float exposure,
	float fee,
	float slip,
	float quantum,
	float fee_per_unit
) {
	int current = a.quantity > 0.0f ? 1 : (a.quantity < 0.0f ? -1 : 0);
	if (target == current) return;
	if (current != 0) trading_fill(a,-a.quantity,price,fee,slip,fee_per_unit);
	a.quantity = 0.0f; a.entry_price = 0.0f; a.age = 0.0f; a.position_peak = 0.0f;
	if (target != 0 && a.equity > 0.0f) {
		float quantity = a.equity * exposure / (price*(1.0f+slip)*(1.0f+fee) + fee_per_unit);
		if (quantum > 0.0f) quantity = floor(quantity / quantum) * quantum;
		quantity *= float(target);
		if (quantity != 0.0f) { a.entry_price = trading_fill(a,quantity,price,fee,slip,fee_per_unit); a.quantity = quantity; }
	}
}

public bool trading_finite(TradingLedger a) {
	return isfinite(a.equity) && isfinite(a.quantity) && isfinite(a.mark)
		&& isfinite(a.peak) && isfinite(a.fees) && isfinite(a.funding)
		&& isfinite(a.slippage) && isfinite(a.fills) && isfinite(a.drawdown) && isfinite(a.entry_price) && isfinite(a.age) && isfinite(a.position_peak)
		&& isfinite(a.pending_target)
		&& isfinite(a.pending_delay)
		&& isfinite(a.pending_limit)
		&& isfinite(a.pending_size)
		&& isfinite(a.liquidations)
		&& isfinite(a.unfilled_iocs)
		&& isfinite(a.initial_margin)
		&& isfinite(a.maintenance_margin)
		&& isfinite(a.maker_pending)
		&& isfinite(a.maker_price)
		&& isfinite(a.maker_deadline)
		&& isfinite(a.maker_requotes)
		&& isfinite(a.maker_fills)
		&& isfinite(a.maker_fallbacks)
		;
}

public void trading_observe(
	RWByteAddressBuffer output,
	RWByteAddressBuffer features,
	uint lane,
	uint row,
	uint window,
	uint width,
	TradingLedger a,
	float capital,
	uint steps,
	uint horizon,
	float maker_window,
	float maker_requote_budget
) {
	uint candle_elements = window * width;
	uint base = lane * (candle_elements + 16u);
	uint start = (row + 1u - window) * width;
	for (uint i=0u; i<candle_elements; ++i) store_f32(output,base+i,load_f32(features,start+i));
	// Signed exposure, equity ratio, peak ratio, drawdown, remaining horizon,
	// unrealized position PnL, holding age and peak position PnL.
	// Giveback is peak minus current PnL; fees/funding stay in net reward.
	store_f32(output,base+candle_elements,a.quantity * a.mark / capital);
	store_f32(output,base+candle_elements+1u,a.equity / capital);
	store_f32(output,base+candle_elements+2u,a.peak / capital);
	store_f32(output,base+candle_elements+3u,(a.peak-a.equity)/a.peak);
	store_f32(output,base+candle_elements+4u,float(horizon-min(steps,horizon))/float(horizon));
	store_f32(output,base+candle_elements+5u,a.quantity * (a.mark - a.entry_price) / capital);
	store_f32(output,base+candle_elements+6u,a.age / float(horizon));
	store_f32(output,base+candle_elements+7u,a.position_peak / capital);
	float pending_delta = a.pending_target == 0.0f ? 0.0f : (float(int(a.pending_target)-2) * a.pending_size - a.quantity);
	store_f32(output,base+candle_elements+8u,pending_delta * a.mark / capital);
	store_f32(output,base+candle_elements+9u,a.pending_delay / float(horizon));
	store_f32(output,base+candle_elements+10u,a.initial_margin / capital);
	store_f32(output,base+candle_elements+11u,a.maintenance_margin / capital);
	// Pending-leg identity, resting-price distance, deadline and chase budget.
	// Flat observations when no maker leg is working.
	bool maker_leg=a.maker_pending!=0.0f;
	store_f32(output,base+candle_elements+12u,maker_leg ? 1.0f : (a.pending_target!=0.0f ? 0.5f : 0.0f));
	store_f32(output,base+candle_elements+13u,maker_leg ? (a.maker_price-a.mark)/a.mark : 0.0f);
	store_f32(output,base+candle_elements+14u,maker_leg ? a.maker_deadline/max(1.0f,maker_window) : 0.0f);
	store_f32(output,base+candle_elements+15u,maker_leg ? a.maker_requotes/max(1.0f,maker_requote_budget) : 0.0f);
}

// Rich execution is a consumer-requested numerical policy extension; no donor
// L2/margin provider exists. Legacy constant-cost arithmetic above is retained.
public float trading_rule(RWByteAddressBuffer rules, uint lane, uint field) {
	return load_f32(rules,lane*40u+field);
}
public float trading_row_mark(RWByteAddressBuffer market, uint row, uint width) {
	return load_f32(market,row*width+(width==3u ? 1u : 3u));
}
public float trading_touch(RWByteAddressBuffer market, uint row, uint width, int side) {
	return load_f32(market,row*width+(width==3u ? 1u : (side>0 ? 46u : 6u)));
}
public void trading_margins(inout TradingLedger a, RWByteAddressBuffer rules, uint lane) {
	float notional=abs(a.quantity)*a.mark;
	a.initial_margin=notional/trading_rule(rules,lane,2u); a.maintenance_margin=0.0f;
	uint tiers=uint(trading_rule(rules,lane,6u));
	for (uint t=0u;t<tiers;++t) {
		uint at=8u+t*4u;
		if (notional<=trading_rule(rules,lane,at) || t+1u==tiers) {
			a.initial_margin=notional*max(1.0f/trading_rule(rules,lane,2u),trading_rule(rules,lane,at+1u));
			a.maintenance_margin=max(0.0f,notional*trading_rule(rules,lane,at+2u)-trading_rule(rules,lane,at+3u)); break;
		}
	}
}
public float trading_entry_size(float equity, float mark, float fill, int side, float exposure, float fee,
	RWByteAddressBuffer rules, uint lane) {
	float cost=max(0.0f,float(side)*(fill-mark))+fill*fee+trading_rule(rules,lane,1u);
	float budget=max(0.0f,equity)*exposure;
	float leverage=trading_rule(rules,lane,2u);
	float result=0.0f;
	uint tiers=uint(trading_rule(rules,lane,6u));
	if (tiers==0u) result=budget/(mark/leverage+cost);
	float lower=0.0f;
	for (uint t=0u;t<tiers;++t) {
		uint at=8u+t*4u;
		float upper=trading_rule(rules,lane,at);
		float initial=max(1.0f/leverage,trading_rule(rules,lane,at+1u));
		float candidate=min(budget/(mark*initial+cost),upper/mark);
		if (candidate*mark>=lower) result=max(result,candidate);
		lower=upper;
	}
	float quantum=trading_rule(rules,lane,0u);
	return quantum>0.0f ? floor(result/quantum)*quantum : result;
}
public void trading_cancel_pending(inout TradingLedger a) {
	a.pending_target=0.0f; a.pending_delay=0.0f; a.pending_limit=0.0f; a.pending_size=0.0f;
	a.maker_pending=0.0f; a.maker_price=0.0f; a.maker_deadline=0.0f; a.maker_requotes=0.0f;
}
// One IOC leg may span levels, but is one filled leg, never one fee per level.
// Visible quantities are shared by close/open legs within this event.
public float trading_depth_fill(inout TradingLedger a, float requested, float mark, float limit,
	RWByteAddressBuffer market, uint row, uint width, RWByteAddressBuffer rules, uint lane,
	float fee, float slip, inout float used_buy, inout float used_sell, out float average) {
	int side=requested>0.0f ? 1 : -1;
	float wanted=abs(requested), filled=0.0f, value=0.0f;
	float consumed=side>0 ? used_buy : used_sell;
	float prefix=0.0f;
	uint count=width==3u ? 1u : uint(load_f32(market,row*width+(side>0 ? 5u : 4u)));
	float quantum=trading_rule(rules,lane,0u);
	for (uint i=0u;i<count && filled<wanted;++i) {
		uint at=row*width+(side>0 ? 46u : 6u)+i*2u;
		float price=width==3u ? load_f32(market,row*width) : load_f32(market,at);
		float capacity=width==3u ? wanted : load_f32(market,at+1u);
		float fill=price*(1.0f+float(side)*slip);
		if (limit>0.0f && (side>0 ? fill>limit : fill<limit)) break;
		float available=max(0.0f,capacity-max(0.0f,consumed-prefix));
		float remaining=wanted-filled;
		float quantity=min(remaining,available);
		// Requests already have whole-lot sizing (or a prior partial-close
		// remainder). Flooring a covered request again can lose a lot to FP32
		// roundoff. Only capacity-limited partial fills need quantization.
		if (quantum>0.0f && available<remaining) quantity=floor(quantity/quantum)*quantum;
		filled=quantity==remaining ? wanted : filled+quantity;
		value+=quantity*fill; prefix+=capacity;
	}
	average=filled>0.0f ? value/filled : 0.0f;
	if (filled>0.0f) {
		float cost=value*fee+filled*trading_rule(rules,lane,1u);
		float drag=float(side)*(value-filled*mark);
		a.equity-=cost+drag; a.fees+=cost; a.slippage+=drag; a.fills+=1.0f;
		if (side>0) used_buy+=filled; else used_sell+=filled;
	}
	return float(side)*filled;
}
public void trading_execute(inout TradingLedger a, int target, float desired, float limit,
	RWByteAddressBuffer market, uint row, uint width, RWByteAddressBuffer rules, uint lane,
	float exposure, float fee, float slip, inout float used_buy, inout float used_sell) {
	int current=a.quantity>0.0f ? 1 : (a.quantity<0.0f ? -1 : 0);
	if (current==target) return;
	float average;
	if (current!=0) {
		float old=a.quantity;
		float closed=trading_depth_fill(a,-old,a.mark,limit,market,row,width,rules,lane,fee,slip,used_buy,used_sell,average);
		a.quantity+=closed;
		if (a.quantity!=0.0f) { a.position_peak*=abs(a.quantity/old); return; }
		a.entry_price=0.0f; a.age=0.0f; a.position_peak=0.0f;
	}
	if (target!=0 && a.equity>0.0f) {
		// Conservative affordability uses the worst visible admissible execution
		// price, not a future quote or assumed unlimited touch liquidity.
		uint count=width==3u ? 1u : uint(load_f32(market,row*width+(target>0 ? 5u : 4u)));
		uint at=row*width+(target>0 ? 46u : 6u)+(count-1u)*2u;
		float worst=width==3u ? load_f32(market,row*width) : load_f32(market,at);
		worst*=1.0f+float(target)*slip;
		if (limit>0.0f) worst=target>0 ? min(worst,limit) : max(worst,limit);
		float quantity=min(desired,trading_entry_size(a.equity,a.mark,worst,target,exposure,fee,rules,lane));
		float opened=trading_depth_fill(a,float(target)*quantity,a.mark,limit,market,row,width,rules,lane,fee,slip,used_buy,used_sell,average);
		if (opened!=0.0f) { a.quantity=opened; a.entry_price=average; a.age=0.0f; a.position_peak=0.0f; }
	}
}

// One observed row against a resting entry leg, matching Trade's strict-through
// oracle: fills only when the executable touch trades strictly through the
// resting price on a later row, never at equality and never partially. The
// deadline counts every observed row including latency-delayed ones; the
// deadline row still checks fills before converting to the IOC leg. Requotes
// only chase a left-behind passive touch and never cross the IOC guard.
public void trading_maker_step(inout TradingLedger a, RWByteAddressBuffer market,
	uint next, uint width, RWByteAddressBuffer rules, uint lane, float exposure,
	float fee, float maker_fee, float book_slip, float maker_requotes,
	inout float used_buy, inout float used_sell) {
	int side=int(a.pending_target)-2;
	float through=trading_touch(market,next,width,side);
	bool filled=side>0 ? through<a.maker_price : through>a.maker_price;
	if (filled && a.pending_size>0.0f) {
		float quantity=float(side)*a.pending_size;
		a.entry_price=trading_fill(a,quantity,a.maker_price,maker_fee,0.0f,trading_rule(rules,lane,1u));
		a.quantity=quantity; a.age=0.0f; a.position_peak=0.0f;
		a.maker_fills+=1.0f;
		trading_cancel_pending(a);
		return;
	}
	if (a.maker_deadline<=1.0f) {
		a.maker_fallbacks+=1.0f;
		a.maker_pending=0.0f;
		float previous_fills=a.fills;
		trading_execute(a,side,a.pending_size,a.pending_limit,market,next,width,rules,lane,exposure,fee,book_slip,used_buy,used_sell);
		int actual=a.quantity>0.0f ? 1 : (a.quantity<0.0f ? -1 : 0);
		if (a.fills==previous_fills || actual!=side || abs(a.quantity)<a.pending_size) a.unfilled_iocs+=1.0f;
		trading_cancel_pending(a);
		return;
	}
	a.maker_deadline-=1.0f;
	float passive=trading_touch(market,next,width,-side);
	bool left_behind=side>0 ? passive>a.maker_price : passive<a.maker_price;
	bool bounded=side>0 ? passive<=a.pending_limit : passive>=a.pending_limit;
	if (left_behind && bounded && a.maker_requotes<maker_requotes) {
		a.maker_price=passive; a.maker_requotes+=1.0f;
	}
}

public bool trading_advance(inout TradingLedger a, int target, RWByteAddressBuffer market,
	uint row, uint next, uint width, RWByteAddressBuffer rules, uint lane, float exposure,
	float fee, float slip, float book_slip, float max_drawdown, float maker_fee,
	float maker_window, float maker_requotes, bool boundary, bool liquidate_boundary,
	out bool terminal) {
	bool legacy=width==3u && trading_rule(rules,lane,2u)==1.0f
		&& trading_rule(rules,lane,3u)==0.0f && trading_rule(rules,lane,6u)==0.0f && trading_rule(rules,lane,4u)==1.0f;
	if (!legacy && a.pending_target==0.0f) {
		int current=a.quantity>0.0f ? 1 : (a.quantity<0.0f ? -1 : 0);
		if (target!=current) {
			int side=target==0 ? -current : target;
			float touch=trading_touch(market,row,width,side);
			a.pending_target=float(target+2); a.pending_delay=trading_rule(rules,lane,3u);
			a.pending_limit=touch*(1.0f+float(side)*trading_rule(rules,lane,4u));
			a.pending_size=target==0 ? 0.0f : trading_entry_size(a.equity,a.mark,touch,side,exposure,fee,rules,lane);
			// Flat-to-position entries under the declared scenario rest at the
			// passive touch; the IOC limit doubles as the requote chase bound.
			if (maker_window>0.0f && width!=3u && current==0 && a.pending_size>0.0f) {
				a.maker_pending=1.0f;
				a.maker_price=trading_touch(market,row,width,-side);
				a.maker_deadline=maker_window;
				a.maker_requotes=0.0f;
			}
		}
	}
	if (a.quantity!=0.0f) a.age+=1.0f;
	float open=load_f32(market,next*width);
	float close=load_f32(market,next*width+1u);
	if (width!=3u) { open=load_f32(market,next*width+3u); close=open; }
	trading_mark(a,open);
	float payment=a.quantity*load_f32(market,next*width+2u);
	a.equity-=payment; a.funding+=payment;
	trading_margins(a,rules,lane);
	bool margin_stop=a.quantity!=0.0f && a.maintenance_margin>0.0f && a.equity<=a.maintenance_margin;
	terminal=trading_risk(a,max_drawdown) || margin_stop;
	// Preserve the original unlevered insolvency rejection. Leveraged gaps can
	// realize a negative equity liability; no silent debt forgiveness/clamping.
	if (legacy && a.equity<=0.0f) return false;
	float used_buy=0.0f, used_sell=0.0f;
	if (legacy) {
		trading_rebalance(a,terminal ? 0 : target,open,exposure,fee,slip,trading_rule(rules,lane,0u),trading_rule(rules,lane,1u));
	} else if (!terminal && a.pending_target!=0.0f) {
		if (a.pending_delay>0.0f) {
			a.pending_delay-=1.0f;
			if (a.maker_pending!=0.0f) a.maker_deadline-=1.0f;
		}
		else if (a.maker_pending!=0.0f)
			trading_maker_step(a,market,next,width,rules,lane,exposure,fee,maker_fee,book_slip,maker_requotes,used_buy,used_sell);
		else {
			float previous_fills=a.fills;
			int requested=int(a.pending_target)-2;
			trading_execute(a,requested,a.pending_size,a.pending_limit,market,next,width,rules,lane,exposure,fee,width==3u ? slip : book_slip,used_buy,used_sell);
			int actual=a.quantity>0.0f ? 1 : (a.quantity<0.0f ? -1 : 0);
			if (a.fills==previous_fills || actual!=requested || (requested!=0 && abs(a.quantity)<a.pending_size)) a.unfilled_iocs+=1.0f;
			trading_cancel_pending(a);
		}
	}
	trading_mark(a,close);
	trading_margins(a,rules,lane);
	margin_stop=margin_stop || (a.quantity!=0.0f && a.maintenance_margin>0.0f && a.equity<=a.maintenance_margin);
	terminal=trading_risk(a,max_drawdown) || terminal || margin_stop;
	if (terminal || boundary) trading_cancel_pending(a);
	if (terminal || (boundary && liquidate_boundary)) {
		if (margin_stop) a.liquidations+=1.0f;
		if (legacy) trading_rebalance(a,0,close,exposure,fee,slip,trading_rule(rules,lane,0u),trading_rule(rules,lane,1u));
		else trading_execute(a,0,0.0f,0.0f,market,next,width,rules,lane,exposure,fee+(margin_stop ? trading_rule(rules,lane,5u) : 0.0f),width==3u ? slip : book_slip,used_buy,used_sell);
		if (a.quantity!=0.0f) return false; // Required close cannot invent depth.
		terminal=trading_risk(a,max_drawdown) || terminal || (boundary && liquidate_boundary);
	}
	trading_margins(a,rules,lane);
	return trading_finite(a);
}
