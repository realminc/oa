//! Consumer-supplied numerical policies; no exchange discovery or authority.
use crate::{Error, Result};

pub(super) const BOOK_LEVELS: usize = 20;
pub(super) const BOOK_MARKET_WIDTH: usize = 6 + BOOK_LEVELS * 4;
pub(super) const MARGIN_TIERS: usize = 8;
pub(super) const RULE_WIDTH: usize = 8 + MARGIN_TIERS * 4;

/// One piecewise-linear margin tier. Rates are fractions of marked notional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradingMarginTier {
	/// Inclusive upper marked-notional limit; the final tier is a hard size cap.
	pub notional_limit: f32,
	/// Initial margin fraction, additionally bounded by requested leverage.
	pub initial_margin_rate: f32,
	pub maintenance_margin_rate: f32,
	/// Maintenance = max(0, notional * rate - deduction).
	pub maintenance_deduction: f32,
}

/// Optional event-aligned visible depth and native mark. No interpolation.
#[derive(Clone, Debug)]
pub struct TradingBook {
	/// Descending bid [price, available base quantity], one to twenty levels.
	pub bids: Vec<[f32; 2]>,
	/// Ascending ask [price, available base quantity], one to twenty levels.
	pub asks: Vec<[f32; 2]>,
	/// Positive mark for equity and margin; distinct from executable bid/ask.
	pub mark_price: f32,
}
impl TradingBook {
	pub(super) fn validate(&self) -> Result<()> {
		if !self.mark_price.is_finite() || !(1e-12..=1e12).contains(&self.mark_price) {
			return Err(Error::invalid_argument("invalid trading book mark"));
		}
		for (levels, bid) in [(&self.bids, true), (&self.asks, false)] {
			if levels.is_empty()
				|| levels.len() > BOOK_LEVELS
				|| levels.iter().any(|v| {
					!v[0].is_finite()
						|| !v[1].is_finite()
						|| !(1e-12..=1e12).contains(&v[0])
						|| !(1e-12..=1e24).contains(&v[1])
				}) || levels.windows(2).any(|w| {
				if bid {
					w[0][0] <= w[1][0]
				} else {
					w[0][0] >= w[1][0]
				}
			}) {
				return Err(Error::invalid_argument(
					"invalid, unsorted or oversized trading depth",
				));
			}
		}
		if self.bids[0][0] >= self.asks[0][0] {
			return Err(Error::invalid_argument("crossed trading book"));
		}
		Ok(())
	}
	pub(super) fn append(&self, row: &[f32; 3], out: &mut Vec<f32>) {
		out.extend_from_slice(row);
		out.extend([
			self.mark_price,
			self.bids.len() as f32,
			self.asks.len() as f32,
		]);
		for levels in [&self.bids, &self.asks] {
			for i in 0..BOOK_LEVELS {
				out.extend(levels.get(i).copied().unwrap_or([0.0; 2]));
			}
		}
	}
}

/// Per-instrument sizing, costs, initial/maintenance margin and IOC policy.
/// Defaults preserve the original continuous, unlevered next-open simulator.
#[derive(Clone, Debug, PartialEq)]
pub struct TradingExecution {
	pub quantity_step: f32,
	pub fee_per_unit: f32,
	/// Requested leverage; 1..=1000. Spot requires one and no margin tiers.
	pub leverage: f32,
	/// Additional observed rows after the original next-row execution boundary.
	pub latency_steps: u32,
	/// IOC price guard relative to executable touch at order submission.
	pub max_slippage_bps: f32,
	/// Additional forced-liquidation commission fraction, not ordinary exits.
	pub liquidation_fee: f32,
	/// Up to eight sorted tiers. Leveraged execution requires explicit tiers.
	pub margin_tiers: Vec<TradingMarginTier>,
}
impl Default for TradingExecution {
	fn default() -> Self {
		Self {
			quantity_step: 0.0,
			fee_per_unit: 0.0,
			leverage: 1.0,
			latency_steps: 0,
			max_slippage_bps: 10000.0,
			liquidation_fee: 0.0,
			margin_tiers: Vec::new(),
		}
	}
}
impl TradingExecution {
	/// # Errors
	/// Rejects nonfinite policies, unsupported ranges, or nonmonotone margin tiers.
	pub fn validate(&self) -> Result<()> {
		if [
			self.quantity_step,
			self.fee_per_unit,
			self.leverage,
			self.max_slippage_bps,
			self.liquidation_fee,
		]
		.iter()
		.any(|v| !v.is_finite())
			|| self.quantity_step < 0.0
			|| self.fee_per_unit < 0.0
			|| !(1.0..=1000.0).contains(&self.leverage)
			|| self.latency_steps > 1_000_000
			|| !(0.0..=10000.0).contains(&self.max_slippage_bps)
			|| !(0.0..0.1).contains(&self.liquidation_fee)
			|| self.margin_tiers.len() > MARGIN_TIERS
			|| (self.leverage > 1.0 && self.margin_tiers.is_empty())
		{
			return Err(Error::invalid_argument("invalid trading execution rules"));
		}
		let mut previous = 0.0;
		let mut previous_initial = 0.0;
		let mut previous_maintenance = 0.0;
		let mut previous_deduction = 0.0;
		for t in &self.margin_tiers {
			if [
				t.notional_limit,
				t.initial_margin_rate,
				t.maintenance_margin_rate,
				t.maintenance_deduction,
			]
			.iter()
			.any(|v| !v.is_finite())
				|| t.notional_limit <= previous
				|| t.notional_limit > 1e24
				|| t.initial_margin_rate <= 0.0
				|| t.initial_margin_rate > 1.0
				|| t.initial_margin_rate < previous_initial
				|| t.maintenance_margin_rate <= 0.0
				|| t.maintenance_margin_rate >= t.initial_margin_rate
				|| t.maintenance_margin_rate < previous_maintenance
				|| t.maintenance_deduction < 0.0
				|| t.maintenance_deduction > previous * t.maintenance_margin_rate
				|| t.maintenance_deduction
					> previous_deduction
						+ previous * (t.maintenance_margin_rate - previous_maintenance)
						+ 1e-5 * (1.0 + previous * t.maintenance_margin_rate)
			{
				return Err(Error::invalid_argument("invalid trading margin tiers"));
			}
			previous = t.notional_limit;
			previous_initial = t.initial_margin_rate;
			previous_maintenance = t.maintenance_margin_rate;
			previous_deduction = t.maintenance_deduction;
		}
		Ok(())
	}
	pub(super) fn append(&self, out: &mut Vec<f32>) {
		out.extend([
			self.quantity_step,
			self.fee_per_unit,
			self.leverage,
			self.latency_steps as f32,
			self.max_slippage_bps / 10000.0,
			self.liquidation_fee,
			self.margin_tiers.len() as f32,
			0.0,
		]);
		for i in 0..MARGIN_TIERS {
			out.extend(self.margin_tiers.get(i).map_or([0.0; 4], |t| {
				[
					t.notional_limit,
					t.initial_margin_rate,
					t.maintenance_margin_rate,
					t.maintenance_deduction,
				]
			}));
		}
	}
}
