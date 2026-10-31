//! Model-independent next-open/event-quote trading research environment.
//!
//! F32 state is an analytical approximation, never an order/account authority.
//! Native funding, contiguous closed-bar features and product policy are supplied
//! by the consumer. Rich execution is analytical, not native exchange authority.
use super::{
	Environment, EnvironmentExecution, EnvironmentSpace, EnvironmentSpec, EnvironmentTransition,
};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{DType, Engine, Error, Matrix, OpAttribute, Result, matrix};

/// Account observation ABI v3: exposure, equity, peak, drawdown, time left,
/// position PnL/age/peak, pending delta/delay and initial/maintenance margin.
pub const TRADING_ACCOUNT_FEATURES: usize = 12;
/// Analytical ledger ABI v3, including entry/peak, pending IOC state and margins.
pub const TRADING_LEDGER_WIDTH: usize = 20;

mod execution;
mod state;
use execution::{BOOK_MARKET_WIDTH, RULE_WIDTH};
pub use execution::{TradingBook, TradingExecution, TradingMarginTier};
pub use state::TradingState;

/// Closed-bar inputs for one contiguous instrument segment, within one split.
#[derive(Clone, Debug)]
pub struct TradingSeries {
	/// Strictly increasing uniformly spaced Unix microseconds.
	pub timestamps: Vec<i64>,
	/// Positive open, positive close, signed native funding rate times oracle.
	/// The coefficient settles old quantity at this row's open; zero is explicit.
	pub market: Vec<[f32; 3]>,
	/// Row-major causal, finite feature rows, already past their warm-up.
	pub features: Vec<f32>,
	/// Positive number of features per closed bar.
	pub feature_width: usize,
	/// Quantity quantum and additive execution commission for this instrument.
	pub execution: TradingExecution,
	/// Optional one book per row, event-aligned with market/funding/features.
	pub books: Vec<TradingBook>,
}

/// Version-one research policy. Fee is a fraction, spread/slippage are basis points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TradingConfig {
	/// Independent device lanes (round-robin over supplied contiguous segments).
	pub environments: u32,
	/// Positive closed-bar window length.
	pub window: u32,
	/// Episode length in bars, 1..=1,000,000; fill counts remain exact in F32.
	pub horizon: u32,
	/// Full 64-bit deterministic reset seed.
	pub seed: u64,
	/// Initial analytical capital per independent lane.
	pub capital: f32,
	/// Fraction of equity budgeted for entry margin and costs, in (0,1].
	/// Instrument execution rules may additionally admit explicit leverage.
	pub exposure: f32,
	/// Actual-notional taker fee fraction on every execution leg.
	pub taker_fee: f32,
	/// Adverse one-way slippage in basis points, excluding spread.
	pub slippage_bps: f32,
	/// Full bid/ask spread in basis points; half is charged per execution leg.
	pub spread_bps: f32,
	/// Peak-equity drawdown fraction that terminates and closes inventory.
	pub max_drawdown: f32,
	/// Perpetuals use short/flat/long; Spot uses flat/long (no short category).
	pub perpetual: bool,
	/// Close and terminate at the horizon; otherwise truncate with final-state bootstrap.
	pub liquidate_at_horizon: bool,
}

impl TradingConfig {
	/// Validate policy before allocations or device work.
	///
	/// # Errors
	/// Rejects nonfinite/unsupported policy or excessive per-step extents.
	pub fn validate(self) -> Result<()> {
		if self.environments == 0
			|| self.environments > 65536
			|| !(1..=1024).contains(&self.window)
			|| self.horizon == 0
			|| self.horizon > 1_000_000
			|| [
				self.capital,
				self.exposure,
				self.taker_fee,
				self.slippage_bps,
				self.spread_bps,
				self.max_drawdown,
			]
			.iter()
			.any(|x| !x.is_finite())
			|| !(0.000001..=1e12).contains(&self.capital)
			|| !(0.000001..=1.0).contains(&self.exposure)
			|| !(0.0..0.1).contains(&self.taker_fee)
			|| !(0.0..1000.0).contains(&self.slippage_bps)
			|| !(0.0..1000.0).contains(&self.spread_bps)
			|| self.max_drawdown <= 0.0
			|| self.max_drawdown >= 1.0
		{
			return Err(Error::invalid_argument(
				"invalid unlevered trading research configuration",
			));
		}
		Ok(())
	}
	fn identity(self) -> u64 {
		[
			self.environments,
			self.window,
			self.horizon,
			self.capital.to_bits(),
			self.exposure.to_bits(),
			self.taker_fee.to_bits(),
			self.slippage_bps.to_bits(),
			self.spread_bps.to_bits(),
			self.max_drawdown.to_bits(),
			u32::from(self.perpetual),
			u32::from(self.liquidate_at_horizon),
		]
		.into_iter()
		.flat_map(u32::to_le_bytes)
		.fold(14695981039346656037_u64, |h, b| {
			(h ^ u64::from(b)).wrapping_mul(1099511628211)
		})
	}
}

/// Device-resident trading dynamics over one Engine and the canonical lifecycle.
/// Observations append signed exposure, equity/capital, peak/capital, drawdown,
/// remaining horizon, unrealized PnL/capital, position age/horizon and peak
/// unrealized position PnL/capital.
#[must_use]
pub struct TradingEnvironment<'engine> {
	engine: &'engine Engine,
	execution: EnvironmentExecution,
	config: TradingConfig,
	spec: EnvironmentSpec,
	market: Matrix,
	features: Matrix,
	bounds: Matrix,
	rules: Matrix,
	ledger: Matrix,
	meta: Matrix,
	done: Matrix,
	observation: Matrix,
	width: u32,
	market_width: u32,
	pending_seed: Option<u64>,
	committed: bool,
	reset_pending: bool,
	rollback_observation: Option<Matrix>,
	force_end: bool,
	tables_valid: bool,
	source: crate::cryptography::Hash,
}

impl<'engine> TradingEnvironment<'engine> {
	/// Change the actual-notional fee between episodes without resetting lane state.
	/// Existing captured programs retain their recorded fee; discard and recapture
	/// the `TradingRolloutProgram` before collecting with the new configuration.
	/// Snapshots remain bound to the fee in effect when they were taken.
	///
	/// # Errors
	/// Rejects invalid fees, active execution, unhealthy state, or any initialized
	/// lane which has advanced beyond reset or retains inventory/pending execution.
	pub fn set_taker_fee(&mut self, taker_fee: f32) -> Result<()> {
		let next = TradingConfig {
			taker_fee,
			..self.config
		};
		next.validate()?;
		if !self.tables_valid
			|| self.execution.has_active_recording()
			|| self.execution.has_pending_event()
		{
			return Err(Error::failed_precondition(
				"fee changes require idle trading execution",
			));
		}
		self.engine.checkpoint()?.wait()?;
		if self.committed {
			self.check_status()?;
			let metadata = self.meta.read::<u32>()?;
			let ledger = self.ledger.read_f32()?;
			if metadata.as_chunks::<4>().0.iter().any(|m| m[1] != 0)
				|| ledger
					.as_chunks::<TRADING_LEDGER_WIDTH>()
					.0
					.iter()
					.any(|a| a[1] != 0.0 || a[12] != 0.0 || a[15] != 0.0)
			{
				return Err(Error::failed_precondition(
					"fee changes require every trading lane at reset",
				));
			}
		}
		self.config = next;
		Ok(())
	}

	/// Validate/upload immutable data and allocate analytical lane state.
	/// Call `reset`, then `submit`/`wait` explicitly before observing it.
	///
	/// # Errors
	/// Rejects discontinuous or invalid data, insufficient episode coverage,
	/// overflowing extents, unsupported policy, allocation or recording failure.
	pub fn new(
		engine: &'engine Engine,
		config: TradingConfig,
		series: &[TradingSeries],
	) -> Result<Self> {
		Self::new_with_row_capacity(engine, config, series, 0)
	}

	/// Allocate stable table capacity for replay across changing contiguous segments.
	/// Padding is unreachable: bounds always describe only supplied source rows.
	/// # Errors
	/// Returns the validation errors of `new` or an excessive capacity error.
	pub fn new_with_row_capacity(
		engine: &'engine Engine,
		config: TradingConfig,
		series: &[TradingSeries],
		capacity: usize,
	) -> Result<Self> {
		let tables = trading_tables(config, series)?;
		let source = state::fingerprint(&tables, series)?;
		let TradingTables {
			mut market,
			mut features,
			bounds,
			rules,
			width,
			market_width,
		} = tables;
		let rows = market.len() / market_width;
		if capacity != 0 && capacity < rows {
			return Err(Error::invalid_argument(
				"trading row capacity is smaller than source rows",
			));
		}
		let capacity = capacity.max(rows);
		if capacity
			.checked_mul(width.max(market_width))
			.is_none_or(|n| n > 50_000_000)
		{
			return Err(Error::resource_exhausted(
				"trading row capacity exceeds 50M elements",
			));
		}
		market.resize(capacity * market_width, 0.);
		features.resize(capacity * width, 0.);
		let lanes = config.environments as usize;
		let observation_width = (config.window as usize)
			.checked_mul(width)
			.and_then(|n| n.checked_add(TRADING_ACCOUNT_FEATURES))
			.ok_or_else(|| Error::invalid_argument("trading observation overflow"))?;
		if lanes
			.checked_mul(observation_width)
			.is_none_or(|n| n > 50_000_000)
		{
			return Err(Error::resource_exhausted(
				"trading observation exceeds 50M elements",
			));
		}

		let spec = EnvironmentSpec::new(
			EnvironmentSpace::continuous(
				"observation",
				[observation_width],
				DType::F32,
				f64::NEG_INFINITY,
				f64::INFINITY,
			)?,
			EnvironmentSpace::discrete("action", if config.perpetual { 3 } else { 2 }, DType::I32)?,
			EnvironmentSpace::continuous("reward", [], DType::F32, f64::NEG_INFINITY, f64::INFINITY)?,
			EnvironmentSpace::binary("terminated", [])?,
			EnvironmentSpace::binary("truncated", [])?,
		)?;
		Ok(Self {
			engine,
			execution: EnvironmentExecution::new(engine),
			config,
			spec,
			market: Matrix::new_from_f32(engine, [market.len() / market_width, market_width], &market)?,
			features: Matrix::new_from_f32(engine, [features.len() / width, width], &features)?,
			bounds: Matrix::new_from_slice(engine, [lanes, 2], &bounds)?,
			rules: Matrix::new_from_f32(engine, [lanes, RULE_WIDTH], &rules)?,
			ledger: Matrix::new_from_f32(
				engine,
				[lanes, TRADING_LEDGER_WIDTH],
				&vec![0.0; lanes * TRADING_LEDGER_WIDTH],
			)?,
			meta: Matrix::new_from_slice(engine, [lanes, 4], &vec![0_u32; lanes * 4])?,
			done: Matrix::new_from_slice(engine, [lanes], &vec![0_u8; lanes])?,
			observation: Matrix::new_from_f32(
				engine,
				[lanes, observation_width],
				&vec![0.0; lanes * observation_width],
			)?,
			width: width as u32,
			market_width: market_width as u32,
			pending_seed: None,
			committed: false,
			reset_pending: false,
			rollback_observation: None,
			force_end: false,
			tables_valid: true,
			source,
		})
	}

	/// Replace resident source tables without replacing captured storage.
	/// Call at a completed rollout boundary; lane state must be reset afterwards.
	/// # Errors
	/// Rejects invalid/discontinuous source data, changed feature width/capacity,
	/// active environment execution, or upload/completion failure.
	pub fn replace_series(&mut self, series: &[TradingSeries]) -> Result<()> {
		if self.execution.has_active_recording() || self.execution.has_pending_event() {
			return Err(Error::failed_precondition(
				"trading table replacement requires idle execution",
			));
		}
		if self.committed
			&& self
				.meta
				.read::<u32>()?
				.as_chunks::<4>()
				.0
				.iter()
				.any(|m| m[1] != 0 && m[1] < self.config.horizon)
		{
			return Err(Error::failed_precondition(
				"cannot replace source inside a continuing episode",
			));
		}
		let tables = trading_tables(self.config, series)?;
		let source = state::fingerprint(&tables, series)?;
		let TradingTables {
			mut market,
			mut features,
			bounds,
			rules,
			width,
			market_width,
		} = tables;
		if width != self.width as usize
			|| market_width != self.market_width as usize
			|| market.len() > self.market.num_elements()
		{
			return Err(Error::invalid_argument(
				"replacement trading table exceeds captured capacity/width",
			));
		}
		market.resize(self.market.num_elements(), 0.);
		features.resize(self.features.num_elements(), 0.);
		self.engine.checkpoint()?.wait()?;
		self.committed = false;
		self.tables_valid = false;
		self.market.write_values(&market)?;
		self.features.write_values(&features)?;
		self.bounds.write_values(&bounds)?;
		self.rules.write_values(&rules)?;
		self.source = source;
		self.tables_valid = true;
		Ok(())
	}
	pub(in crate::ml) fn replay_initialized(&self) -> bool {
		self.committed && self.tables_valid
	}

	pub(in crate::ml) fn replay_tables_valid(&self) -> bool {
		self.tables_valid
	}

	pub(in crate::ml) fn replay_slots(&self) -> Vec<&Matrix> {
		vec![&self.ledger, &self.meta, &self.done, &self.observation]
	}

	pub(in crate::ml) fn replay_identity(&self) -> Vec<u64> {
		vec![
			self.market.value_id(),
			self.features.value_id(),
			self.bounds.value_id(),
			self.rules.value_id(),
			self.ledger.value_id(),
			self.meta.value_id(),
			self.done.value_id(),
		]
	}
	pub(in crate::ml) fn replay_config(&self) -> TradingConfig {
		self.config
	}
	pub(in crate::ml) fn replay_observation(&mut self, value: Matrix) {
		self.observation = value;
		self.rollback_observation = None;
		self.reset_pending = false;
		self.committed = true;
	}

	/// Step and close all lanes at this collection boundary, charging exit costs
	/// in this transition. This is a task terminal, never a free reset.
	///
	/// # Errors
	/// Returns the normal Environment step validation/recording failures.
	pub fn step_terminal(&mut self, action: &Matrix) -> Result<EnvironmentTransition> {
		self.force_end = true;
		let result = self.step(action);
		self.force_end = false;
		result
	}

	/// Analytical rows: equity, quantity, mark, peak, fees, funding, slippage,
	/// fill count, maximum drawdown, entry price, age in source steps and peak
	/// unrealized position PnL. Peak is zero while flat or never profitable.
	/// These are never exact money values.
	pub const fn ledger(&self) -> &Matrix {
		&self.ledger
	}

	/// Cursor, episode steps, episode ordinal, failure status (zero means healthy).
	pub const fn metadata(&self) -> &Matrix {
		&self.meta
	}

	/// Explicitly observe status after exact completion and reject any failed lane.
	/// No action/price readback occurs during device rollout steps.
	///
	/// # Errors
	/// Returns readback/completion failure or a poisoned numerical/action lane.
	pub fn check_status(&self) -> Result<()> {
		if !self.committed
			|| self.execution.has_active_recording()
			|| self.execution.has_pending_event()
		{
			return Err(Error::failed_precondition(
				"trading status requires completed submitted state",
			));
		}
		if self
			.meta
			.read::<u32>()?
			.as_chunks::<4>()
			.0
			.iter()
			.any(|r| r[3] != 0)
		{
			return Err(Error::failed_precondition(
				"trading lane rejected action, insolvency, nonfinite arithmetic or exhausted episode counter",
			));
		}
		Ok(())
	}

	fn reset_commands(&mut self, only_done: bool) -> Result<()> {
		self.reset_commands_impl(only_done, None)
	}
	pub(in crate::ml) fn reset_replay_commands(
		&mut self,
		only_done: bool,
		cursor: &Matrix,
	) -> Result<()> {
		self.reset_commands_impl(only_done, Some(cursor))
	}
	fn reset_commands_impl(&mut self, only_done: bool, cursor: Option<&Matrix>) -> Result<()> {
		if !self.tables_valid {
			return Err(Error::failed_precondition(
				"trading tables require a successful replacement",
			));
		}
		if only_done && !self.committed && !self.reset_pending {
			return Err(Error::failed_precondition(
				"trading requires full reset after construction/cancellation",
			));
		}
		if self.rollback_observation.is_none() {
			self.rollback_observation = Some(self.observation.clone());
		}
		if cursor.is_none() {
			self.observation = matrix::copy(&self.observation)?;
		}
		let seed = self.pending_seed.unwrap_or(self.config.seed);
		let mut inputs = vec![
			&self.market,
			&self.features,
			&self.bounds,
			&self.ledger,
			&self.meta,
			&self.done,
			&self.observation,
		];
		let mut buffers = vec![
			BufferBinding::read(self.market.storage()),
			BufferBinding::read(self.features.storage()),
			BufferBinding::read(self.bounds.storage()),
			BufferBinding::read_write(self.ledger.storage()),
			BufferBinding::read_write(self.meta.storage()),
			BufferBinding::read_write(self.done.storage()),
			BufferBinding::read_write(self.observation.storage()),
		];
		let mut pushes = vec![
			PushConstant::U32(self.config.environments),
			PushConstant::U32(self.config.window),
			PushConstant::U32(self.width),
			PushConstant::U32(self.config.horizon),
			PushConstant::U32(seed as u32),
			PushConstant::U32((seed >> 32) as u32),
			PushConstant::U32(u32::from(only_done)),
			PushConstant::F32(self.config.capital),
			PushConstant::U32(self.market_width),
		];
		if let Some(cursor) = cursor {
			inputs.insert(3, cursor);
			buffers.push(BufferBinding::read(cursor.storage()));
			pushes.drain(4..6);
		}
		let mut attributes = vec![OpAttribute::UnsignedInteger {
			name: "config_identity".into(),
			value: self.config.identity(),
		}];
		if cursor.is_none() {
			attributes.push(OpAttribute::UnsignedInteger {
				name: "seed".into(),
				value: seed,
			});
		}
		attributes.push(OpAttribute::Boolean {
			name: "only_completed".into(),
			value: only_done,
		});

		self.market.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: if cursor.is_some() {
					KernelId::MlTradingResetCursorF32
				} else {
					KernelId::MlTradingResetF32
				},
				buffers: &buffers,
				push_constants: &pushes,
				workgroups: KernelId::MlTradingResetF32.linear_workgroups(self.config.environments),
			},
			SemanticDispatch {
				contract: if cursor.is_some() {
					crate::core::operation::ml::TRADING_RESET_CURSOR
				} else {
					crate::core::operation::ml::TRADING_RESET
				},
				inputs: &inputs,
				outputs: &[&self.ledger, &self.meta, &self.done, &self.observation],
				attributes: &attributes,
			},
		)?;
		if !only_done {
			self.reset_pending = true;
		}
		Ok(())
	}
	pub(in crate::ml) fn step_replay_commands(
		&mut self,
		action: &Matrix,
		cursor: &Matrix,
	) -> Result<EnvironmentTransition> {
		self.step_commands(action, Some(cursor))
	}
	fn step_commands(
		&mut self,
		action: &Matrix,
		cursor: Option<&Matrix>,
	) -> Result<EnvironmentTransition> {
		self
			.spec
			.validate_action(action, self.config.environments)?;
		if !self.committed && !self.reset_pending {
			return Err(Error::failed_precondition("trading step requires reset"));
		}
		if !self.engine.owns_matrix(action) {
			return Err(Error::invalid_argument(
				"trading action belongs to another Engine",
			));
		}
		// Immutable snapshots survive later reset/step calls in the same transaction.
		let before = if cursor.is_some() {
			self.observation.clone()
		} else {
			matrix::copy(&self.observation)?
		};
		let lanes = self.config.environments as usize;
		let next = Matrix::allocate(
			self.market.engine_handle(),
			self.observation.shape().to_vec(),
			self.observation.num_elements(),
			DType::F32,
		)?;
		let reward = Matrix::allocate(self.market.engine_handle(), vec![lanes], lanes, DType::F32)?;
		let terminated = Matrix::new_from_slice(self.engine, [lanes], &vec![0_u8; lanes])?;
		let truncated = Matrix::new_from_slice(self.engine, [lanes], &vec![0_u8; lanes])?;
		let mut buffers = vec![
			BufferBinding::read(self.market.storage()),
			BufferBinding::read(self.features.storage()),
			BufferBinding::read(self.bounds.storage()),
			BufferBinding::read(action.storage()),
			BufferBinding::read_write(self.ledger.storage()),
			BufferBinding::read_write(self.meta.storage()),
			BufferBinding::read_write(self.done.storage()),
			BufferBinding::write(next.storage()),
			BufferBinding::write(reward.storage()),
			BufferBinding::write(terminated.storage()),
			BufferBinding::write(truncated.storage()),
		];
		let c = self.config;
		let mut pushes = vec![
			PushConstant::U32(c.environments),
			PushConstant::U32(c.window),
			PushConstant::U32(self.width),
			PushConstant::U32(c.horizon),
			PushConstant::U32(u32::from(c.perpetual)),
			PushConstant::U32(u32::from(c.liquidate_at_horizon)),
			PushConstant::U32(u32::from(self.force_end)),
			PushConstant::F32(c.capital),
			PushConstant::F32(c.exposure),
			PushConstant::F32(c.taker_fee),
			PushConstant::F32((c.slippage_bps + c.spread_bps * 0.5) / 10000.0),
			PushConstant::F32(c.max_drawdown),
			PushConstant::U32(self.market_width),
			PushConstant::F32(c.slippage_bps / 10000.0),
		];
		if let Some(cursor) = cursor {
			buffers.push(BufferBinding::read(cursor.storage()));
			pushes.remove(6);
		}
		buffers.push(BufferBinding::read(self.rules.storage()));
		let mut inputs = vec![
			&self.market,
			&self.features,
			&self.bounds,
			action,
			&self.rules,
			&self.ledger,
			&self.meta,
			&self.done,
			&next,
			&reward,
			&terminated,
			&truncated,
		];
		if let Some(cursor) = cursor {
			inputs.insert(4, cursor);
		}
		let mut attributes = vec![OpAttribute::UnsignedInteger {
			name: "config_identity".into(),
			value: c.identity(),
		}];
		if cursor.is_none() {
			attributes.push(OpAttribute::Boolean {
				name: "force_end".into(),
				value: self.force_end,
			});
		}

		self.market.engine_handle().record_semantic(
			ComputeDispatch {
				kernel: if cursor.is_some() {
					KernelId::MlTradingStepCursorF32
				} else {
					KernelId::MlTradingStepF32
				},
				buffers: &buffers,
				push_constants: &pushes,
				workgroups: KernelId::MlTradingStepF32.linear_workgroups(c.environments),
			},
			SemanticDispatch {
				contract: if cursor.is_some() {
					crate::core::operation::ml::TRADING_STEP_CURSOR
				} else {
					crate::core::operation::ml::TRADING_STEP
				},
				inputs: &inputs,
				outputs: &[
					&self.ledger,
					&self.meta,
					&self.done,
					&next,
					&reward,
					&terminated,
					&truncated,
				],
				attributes: &attributes,
			},
		)?;
		if self.rollback_observation.is_none() {
			self.rollback_observation = Some(self.observation.clone());
		}
		self.observation = next.clone();
		Ok(EnvironmentTransition::new(
			before, next, reward, terminated, truncated,
		))
	}
}

impl Environment for TradingEnvironment<'_> {
	fn engine(&self) -> &Engine {
		self.engine
	}

	fn spec(&self) -> &EnvironmentSpec {
		&self.spec
	}

	fn environments(&self) -> u32 {
		self.config.environments
	}

	fn observation(&self) -> &Matrix {
		&self.observation
	}

	fn execution(&self) -> &EnvironmentExecution {
		&self.execution
	}

	fn execution_mut(&mut self) -> &mut EnvironmentExecution {
		&mut self.execution
	}

	fn record_reset(&mut self, seed: u64) -> Result<()> {
		self.pending_seed = Some(seed);
		self.reset_commands(false)
	}

	fn record_reset_completed(&mut self) -> Result<()> {
		self.reset_commands(true)
	}

	fn commit_recorded_state(&mut self) {
		if let Some(seed) = self.pending_seed.take() {
			self.config.seed = seed;
		}
		self.rollback_observation = None;
		self.committed |= self.reset_pending;
		self.reset_pending = false;
	}

	fn rollback_recorded_state(&mut self) {
		if let Some(observation) = self.rollback_observation.take() {
			self.observation = observation;
		}
		self.pending_seed = None;
		self.reset_pending = false;
	}

	fn record_step(&mut self, action: &Matrix) -> Result<EnvironmentTransition> {
		self.step_commands(action, None)
	}
}

struct TradingTables {
	market: Vec<f32>,
	features: Vec<f32>,
	bounds: Vec<u32>,
	rules: Vec<f32>,
	width: usize,
	market_width: usize,
}
fn trading_tables(config: TradingConfig, series: &[TradingSeries]) -> Result<TradingTables> {
	config.validate()?;
	if series.is_empty() || series.len() > config.environments as usize {
		return Err(Error::invalid_argument(
			"trading requires at least one lane per contiguous segment",
		));
	}
	let width = series[0].feature_width;
	if width == 0 || width > 1024 {
		return Err(Error::invalid_argument("invalid trading feature width"));
	}
	let market_width = if series.iter().any(|s| !s.books.is_empty()) {
		BOOK_MARKET_WIDTH
	} else {
		3
	};
	if market_width != 3 && series.iter().any(|s| s.books.is_empty()) {
		return Err(Error::invalid_argument(
			"mixed legacy/book execution series require explicit books for every series",
		));
	}
	let mut market = Vec::new();
	let mut features = Vec::new();
	let mut segments = Vec::new();
	for s in series {
		s.execution.validate()?;
		let n = s.market.len();
		if !config.perpetual && (s.execution.leverage != 1.0 || !s.execution.margin_tiers.is_empty()) {
			return Err(Error::invalid_argument(
				"Spot cannot use leveraged margin rules",
			));
		}
		if !s.books.is_empty() && s.books.len() != n {
			return Err(Error::invalid_argument(
				"trading book row coverage mismatch",
			));
		}
		for book in &s.books {
			book.validate()?;
		}
		if s.feature_width != width
			|| s.timestamps.len() != n
			|| n < config.window as usize + config.horizon as usize
			|| n.checked_mul(width) != Some(s.features.len())
			|| s.features.iter().any(|x| !x.is_finite())
			|| s.market.iter().any(|x| {
				x.iter().any(|v| !v.is_finite())
					|| !(1e-12..=1e12).contains(&x[0])
					|| !(1e-12..=1e12).contains(&x[1])
					|| (!config.perpetual && x[2] != 0.0)
			}) {
			return Err(Error::invalid_argument(
				"invalid trading series shape, prices, funding or features",
			));
		}
		let interval = s.timestamps[1]
			.checked_sub(s.timestamps[0])
			.filter(|d| *d > 0)
			.ok_or_else(|| Error::invalid_argument("invalid trading time interval"))?;
		if s
			.timestamps
			.windows(2)
			.any(|w| w[1].checked_sub(w[0]) != Some(interval))
		{
			return Err(Error::invalid_argument(
				"trading episode crosses a timestamp gap",
			));
		}
		let offset = market.len() / market_width;
		let end = offset
			.checked_add(n)
			.ok_or_else(|| Error::invalid_argument("trading row extent overflow"))?;
		if end
			.checked_mul(width.max(market_width))
			.is_none_or(|x| x > 50_000_000)
		{
			return Err(Error::resource_exhausted(
				"trading upload exceeds 50M-element budget",
			));
		}
		segments.push([
			u32::try_from(offset + config.window as usize - 1)
				.map_err(|_| Error::invalid_argument("trading offset overflow"))?,
			u32::try_from(end - 1).map_err(|_| Error::invalid_argument("trading end overflow"))?,
		]);
		if s.books.is_empty() {
			market.extend(s.market.iter().flatten().copied());
		} else {
			for (row, book) in s.market.iter().zip(&s.books) {
				book.append(row, &mut market);
			}
		}
		features.extend_from_slice(&s.features);
	}
	let bounds = (0..config.environments as usize)
		.flat_map(|i| segments[i % segments.len()])
		.collect();
	let mut rules = Vec::with_capacity(config.environments as usize * RULE_WIDTH);
	for i in 0..config.environments as usize {
		series[i % series.len()].execution.append(&mut rules);
	}
	Ok(TradingTables {
		market,
		features,
		bounds,
		rules,
		width,
		market_width,
	})
}
