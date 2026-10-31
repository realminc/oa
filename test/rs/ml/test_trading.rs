use oa::Matrix;
use oa::ml::environment::{Environment, TradingConfig, TradingEnvironment, TradingSeries};
fn config() -> TradingConfig {
	TradingConfig {
		environments: 3,
		window: 2,
		horizon: 3,
		seed: 0x1122334455667788,
		capital: 100.0,
		exposure: 0.5,
		taker_fee: 0.001,
		slippage_bps: 2.0,
		spread_bps: 4.0,
		max_drawdown: 0.5,
		perpetual: true,
		liquidate_at_horizon: true,
	}
}
fn data() -> TradingSeries {
	TradingSeries {
		timestamps: (0..20).map(|i| i * 60_000_000).collect(),
		market: vec![[100.0, 100.0, 0.0]; 20],
		features: vec![0.25; 20],
		feature_width: 1,
		execution: Default::default(),
		books: Vec::new(),
	}
}
#[test]
fn trading_configuration_rejects_invalid_costs() {
	config().validate().unwrap();
	for c in [
		TradingConfig {
			exposure: 1.01,
			..config()
		},
		TradingConfig {
			taker_fee: f32::NAN,
			..config()
		},
		TradingConfig {
			horizon: 0,
			..config()
		},
		TradingConfig {
			spread_bps: -1.0,
			..config()
		},
	] {
		assert!(c.validate().is_err());
	}
}
test_vk!(
	trading_fee_change_preserves_reset_state_and_rejects_active_episodes,
	engine,
	{
		let mut e = TradingEnvironment::new(&engine, config(), &[data()])?;
		e.set_taker_fee(0.00015)?;
		e.reset(config().seed)?;
		assert!(e.set_taker_fee(0.0005).is_err());
		let event = e.submit()?;
		assert!(e.set_taker_fee(0.0005).is_err());
		e.wait(&event)?;
		let before = e.snapshot()?;
		let meta = e.metadata().read::<u32>()?;
		let ledger = e.ledger().read_f32()?;
		let observation = e.observation().read_f32()?;
		for fee in [-0.1, f32::NAN, f32::INFINITY, 0.1] {
			assert!(e.set_taker_fee(fee).is_err());
		}
		e.set_taker_fee(0.0005)?;
		assert_eq!(meta, e.metadata().read::<u32>()?);
		assert_eq!(ledger, e.ledger().read_f32()?);
		assert_eq!(observation, e.observation().read_f32()?);
		assert!(e.restore(&before).is_err());
		let after = e.snapshot()?;
		e.restore(&after)?;
		let action = Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 1])?;
		e.step(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		assert!(e.set_taker_fee(0.00015).is_err());
		// Independent constant-price entry: 50 USD budget includes 4 bps adverse
		// execution and 5 bps commission. The third lane remains flat.
		let state = e.ledger().read_f32()?;
		let expected_fee = 50.0 * 0.0005 / 1.0005;
		assert!((state[4] - expected_fee).abs() < 1e-6);
		assert!((state[24] - expected_fee * 0.9996 / 1.0004).abs() < 1e-6);
		assert_eq!(state[44], 0.0);
		e.step(&action)?;
		e.step(&action)?;
		e.reset_completed()?;
		let event = e.submit()?;
		e.wait(&event)?;
		let meta = e.metadata().read::<u32>()?;
		e.set_taker_fee(0.00015)?;
		assert_eq!(meta, e.metadata().read::<u32>()?);
		assert_eq!(meta[2], 1);
		Ok(())
	}
);
test_vk!(
	trading_lifecycle_costs_final_observation_and_rollback,
	engine,
	{
		let mut e = TradingEnvironment::new(&engine, config(), &[data()])?;
		assert!(e.check_status().is_err());
		e.reset(config().seed)?;
		let event = e.submit()?;
		assert!(e.begin().is_err());
		e.wait(&event)?;
		e.check_status()?;
		let initial = e.observation().read_f32()?;
		let initial_meta = e.metadata().read::<u32>()?;
		let action = Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 1])?;
		let cancelled = e.step(&action)?;
		e.cancel()?;
		assert_eq!(initial, e.observation().read_f32()?);
		assert_eq!(initial_meta, e.metadata().read::<u32>()?);
		drop(cancelled);
		let first = e.step(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		let state = e.ledger().read_f32()?;
		assert!(state[0] < 100.0 && state[1] > 0.0);
		assert!(state[20] < 100.0 && state[21] < 0.0);
		assert_eq!(state[40], 100.0);
		assert_eq!(first.observation().read_f32()?, initial);
		let saved_next = first.next_observation().read_f32()?;
		let second = e.step(&action)?;
		let third = e.step(&action)?;
		e.reset_completed()?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		assert_eq!(third.terminated().read::<u8>()?, [1, 1, 1]);
		assert_eq!(third.truncated().read::<u8>()?, [0, 0, 0]);
		assert_eq!(second.reward().read_f32()?, [0.0; 3]);
		let final_obs = third.next_observation().read_f32()?;
		assert_eq!(final_obs[2], 0.0);
		assert_eq!(final_obs[6], 0.0); // flat, zero horizon.
		assert_eq!(first.next_observation().read_f32()?, saved_next);
		assert_eq!(
			e.ledger().read_f32()?,
			[
				100.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0
			]
			.repeat(3)
		);
		let meta = e.metadata().read::<u32>()?;
		assert!(
			meta
				.as_chunks::<4>()
				.0
				.iter()
				.all(|r| r[1] == 0 && r[2] == 1 && r[3] == 0)
		);
		// Repeat full reset with exact seed, including its high word.
		e.reset(config().seed)?;
		let event = e.submit()?;
		e.wait(&event)?;
		assert_eq!(e.metadata().read::<u32>()?, initial_meta);
		e.close()?;
		Ok(())
	}
);
test_vk!(
	trading_truncation_bootstraps_and_spot_rejects_short_category,
	engine,
	{
		let mut c = config();
		c.liquidate_at_horizon = false;
		c.horizon = 1;
		c.perpetual = false;
		let mut source = data();
		source.market.fill([100.0, 110.0, 0.0]);
		let mut e = TradingEnvironment::new(&engine, c, &[source])?;
		e.reset(c.seed)?;
		let event = e.submit()?;
		e.wait(&event)?;
		let action = Matrix::new_from_slice(&engine, [3], &[1_i32; 3])?;
		let step = e.step(&action)?;
		e.reset_completed()?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		assert_eq!(step.terminated().read::<u8>()?, [0; 3]);
		assert_eq!(step.truncated().read::<u8>()?, [1; 3]);
		assert!(step.next_observation().read_f32()?[2] > 0.0);
		assert!(step.next_observation().read_f32()?[9] > 0.0); // Preserve the position peak for bootstrap.
		assert_eq!(e.ledger().read_f32()?[11], 0.0); // Reset state is independent of the saved final observation.
		let bad = Matrix::new_from_slice(&engine, [3], &[2_i32, 1, 0])?;
		e.step(&bad)?;
		let event = e.submit()?;
		e.wait(&event)?;
		assert!(e.check_status().is_err());
		assert_eq!(e.metadata().read::<u32>()?[3], 1);
		e.close()?;
		Ok(())
	}
);

test_vk!(
	trading_resets_only_risk_stopped_lanes_and_closes_collection_boundary,
	engine,
	{
		let mut c = config();
		c.max_drawdown = 0.05;
		let mut d = data();
		d.market = vec![[100.0, 100.0, 0.0]; 5];
		d.market[2] = [100.0, 80.0, 0.0];
		d.timestamps.truncate(5);
		d.features.truncate(5);
		let mut e = TradingEnvironment::new(&engine, c, &[d])?;
		e.reset(c.seed)?;
		let event = e.submit()?;
		e.wait(&event)?;
		let action = Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 1])?;
		let step = e.step(&action)?;
		e.reset_completed()?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		assert_eq!(step.terminated().read::<u8>()?, [1, 0, 0]);
		let meta = e.metadata().read::<u32>()?;
		assert_eq!([meta[1], meta[5], meta[9]], [0, 1, 1]);
		assert_eq!([meta[2], meta[6], meta[10]], [1, 0, 0]);
		assert!(e.ledger().read_f32()?[21] < 0.0);
		let final_step = e.step_terminal(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		assert_eq!(final_step.terminated().read::<u8>()?, [1, 1, 1]);
		assert_eq!(final_step.truncated().read::<u8>()?, [0, 0, 0]);
		let ledger = e.ledger().read_f32()?;
		assert_eq!([ledger[1], ledger[21], ledger[41]], [0.0, 0.0, 0.0]);
		e.close()?;
		Ok(())
	}
);

// New execution and position-state branch: independent exact scalar expectations.
test_vk!(
	trading_contract_fees_quantization_and_position_observations,
	engine,
	{
		use oa::ml::environment::TradingExecution;
		let mut c = config();
		c.capital = 1000.0;
		c.exposure = 1.0;
		c.slippage_bps = 0.0;
		c.spread_bps = 0.0;
		c.horizon = 4;
		let mut source = data();
		source.execution = TradingExecution {
			quantity_step: 3.0,
			fee_per_unit: 0.04,
			..Default::default()
		};
		let mut e = TradingEnvironment::new(&engine, c, &[source])?;
		e.reset(c.seed)?;
		let event = e.submit()?;
		e.wait(&event)?;
		assert_eq!(e.observation().shape(), [3, 14]);
		let action = Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 1])?;
		let first = e.step(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		let ledger = e.ledger().read_f32()?;
		assert_eq!(ledger[1], 9.0);
		assert_eq!(ledger[21], -9.0);
		assert!((ledger[4] - 1.26).abs() < 1e-5);
		assert_eq!(ledger[9], 100.0);
		assert_eq!(ledger[10], 0.0);
		let obs = first.next_observation().read_f32()?;
		assert_eq!(obs[7], 0.0);
		assert_eq!(obs[8], 0.0);
		let held = e.step(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		assert_eq!(held.next_observation().read_f32()?[8], 0.25);
		assert_eq!(e.ledger().read_f32()?[7], 1.0);
		let end = e.step_terminal(&action)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		let ledger = e.ledger().read_f32()?;
		assert_eq!(ledger[1], 0.0);
		assert_eq!(ledger[7], 2.0);
		assert!((ledger[4] - 2.52).abs() < 1e-5);
		assert_eq!(ledger[9], 0.0);
		assert_eq!(ledger[10], 0.0);
		assert_eq!(end.next_observation().read_f32()?[8], 0.0);
		e.close()?;
		Ok(())
	}
);

// Independent long/short price path: the position peak survives giveback,
// but never survives closing, reversing or resetting the position.
test_vk!(
	trading_position_peak_is_causal_and_resets_per_position,
	engine,
	{
		let mut c = config();
		c.capital = 1000.0;
		c.exposure = 1.0;
		c.horizon = 5;
		c.taker_fee = 0.0;
		c.slippage_bps = 0.0;
		c.spread_bps = 0.0;
		let source = TradingSeries {
			timestamps: (0..7).map(|i| i * 1_000_000).collect(),
			market: vec![
				[100.0, 100.0, 0.0],
				[100.0, 100.0, 0.0],
				[100.0, 110.0, 0.0],
				[110.0, 90.0, 0.0],
				[90.0, 95.0, 0.0],
				[95.0, 100.0, 0.0],
				[100.0, 100.0, 0.0],
			],
			features: vec![0.0; 7],
			feature_width: 1,
			execution: Default::default(),
			books: Vec::new(),
		};
		let mut e = TradingEnvironment::new(&engine, c, &[source])?;
		e.reset(c.seed)?;
		let event = e.submit()?;
		e.wait(&event)?;
		let hold = Matrix::new_from_slice(&engine, [3], &[2_i32, 0, 1])?;
		for (index, expected_peak, expected_pnl, expected_reward) in [
			(0, [0.1, 0.0, 0.0], [0.1, -0.1, 0.0], [0.1, -0.1, 0.0]),
			(1, [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0], [-0.2, 0.2, 0.0]),
			(2, [0.1, 0.1, 0.0], [-0.05, 0.05, 0.0], [0.05, -0.05, 0.0]),
		] {
			let step = e.step(&hold)?;
			let event = e.submit()?;
			e.wait(&event)?;
			e.check_status()?;
			let obs = step.next_observation().read_f32()?;
			let reward = step.reward().read_f32()?;
			let ledger = e.ledger().read_f32()?;
			for lane in 0..3 {
				assert!(
					(obs[lane * 14 + 9] - expected_peak[lane]).abs() < 1e-6,
					"peak step {index} lane {lane}"
				);
				assert!((obs[lane * 14 + 7] - expected_pnl[lane]).abs() < 1e-6);
				assert!((ledger[lane * 20 + 11] / c.capital - expected_peak[lane]).abs() < 1e-6);
				assert!(
					(reward[lane] - expected_reward[lane]).abs() < 1e-6,
					"reward changed by observation state"
				);
			}
		}
		let reverse = Matrix::new_from_slice(&engine, [3], &[0_i32, 2, 1])?;
		let step = e.step(&reverse)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		let obs = step.next_observation().read_f32()?;
		assert_eq!(obs[9], 0.0); // New short is losing: old long peak is gone.
		assert_eq!(obs[8], 0.0); // New entry age.
		assert!((obs[23] - obs[21]).abs() < 1e-6); // New long peak is its own PnL.
		let step = e.step(&reverse)?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		assert_eq!(step.terminated().read::<u8>()?, [1; 3]);
		let final_obs = step.next_observation().read_f32()?;
		assert_eq!([final_obs[9], final_obs[23], final_obs[37]], [0.0; 3]);
		e.reset_completed()?;
		let event = e.submit()?;
		e.wait(&event)?;
		e.check_status()?;
		let ledger = e.ledger().read_f32()?;
		assert_eq!([ledger[11], ledger[31], ledger[51]], [0.0; 3]);
		e.close()?;
		Ok(())
	}
);

#[test]
fn trading_execution_rules_reject_nonfinite_and_negative_values() {
	use oa::ml::environment::TradingExecution;
	assert!(TradingExecution::default().validate().is_ok());
	for (quantity_step, fee_per_unit) in [
		(-1.0, 0.0),
		(0.0, -1.0),
		(f32::NAN, 0.0),
		(0.0, f32::INFINITY),
	] {
		assert!(
			TradingExecution {
				quantity_step,
				fee_per_unit,
				..Default::default()
			}
			.validate()
			.is_err()
		);
	}
}

fn event_config(horizon: u32) -> TradingConfig {
	TradingConfig {
		environments: 1,
		window: 2,
		horizon,
		seed: 1,
		capital: 1000.0,
		exposure: 1.0,
		taker_fee: 0.0,
		slippage_bps: 0.0,
		spread_bps: 0.0,
		max_drawdown: 0.99,
		perpetual: true,
		liquidate_at_horizon: true,
	}
}
fn event_series(horizon: u32) -> TradingSeries {
	let n = horizon as usize + 2;
	TradingSeries {
		timestamps: (0..n).map(|i| i as i64 * 1_000_000).collect(),
		market: vec![[100.0, 100.0, 0.0]; n],
		features: vec![0.0; n],
		feature_width: 1,
		execution: Default::default(),
		books: Vec::new(),
	}
}

test_vk!(
	trading_visible_depth_partial_reversal_and_exact_leg_costs,
	engine,
	{
		use oa::ml::environment::TradingBook;
		let mut c = event_config(5);
		c.taker_fee = 0.001;
		let mut source = event_series(c.horizon);
		source.execution.quantity_step = 1.0;
		source.execution.fee_per_unit = 0.02;
		source.books = vec![
			TradingBook {
				bids: vec![[99.0, 3.0]],
				asks: vec![[101.0, 2.0], [102.0, 3.0]],
				mark_price: 100.0
			};
			7
		];
		let mut env = TradingEnvironment::new(&engine, c, &[source])?;
		env.reset(c.seed)?;
		let event = env.submit()?;
		env.wait(&event)?;
		for (action, quantity, equity, fills, fees) in [
			(2_i32, 5.0, 991.392, 1.0, 0.608),
			(0, 2.0, 988.035, 2.0, 0.965),
			(1, 0.0, 985.797, 3.0, 1.203),
		] {
			let step = env.step(&Matrix::new_from_slice(&engine, [1], &[action])?)?;
			let event = env.submit()?;
			env.wait(&event)?;
			env.check_status()?;
			let state = env.ledger().read_f32()?;
			assert_eq!(state[1], quantity);
			assert_eq!(state[7], fills);
			assert!(
				(state[0] - equity).abs() < 0.001,
				"equity {} != {equity}",
				state[0]
			);
			assert!((state[4] - fees).abs() < 1e-5);
			assert!(step.reward().read_f32()?[0] <= 0.0);
		}
		assert_eq!(env.ledger().read_f32()?[17], 2.0);
		env.close()?;
		Ok(())
	}
);

test_vk!(trading_delayed_target_and_portable_state_resume, engine, {
	use oa::ml::environment::TradingState;
	let c = event_config(5);
	let mut source = event_series(c.horizon);
	source.execution.latency_steps = 1;
	source.execution.quantity_step = 1.0;
	let mut first = TradingEnvironment::new(&engine, c, &[source.clone()])?;
	first.reset(c.seed)?;
	let event = first.submit()?;
	first.wait(&event)?;
	first.step(&Matrix::new_from_slice(&engine, [1], &[2_i32])?)?;
	let event = first.submit()?;
	first.wait(&event)?;
	first.check_status()?;
	let ledger = first.ledger().read_f32()?;
	assert_eq!(ledger[1], 0.0);
	assert_eq!(ledger[12], 3.0);
	assert_eq!(ledger[13], 0.0);
	let snapshot = first.snapshot()?;
	let bytes = snapshot.to_bytes();
	let decoded = TradingState::new_from_bytes(&bytes)?;
	let mut corrupt = bytes.clone();
	corrupt[100] ^= 1;
	assert!(TradingState::new_from_bytes(&corrupt).is_err());
	assert!(first.replace_series(&[source.clone()]).is_err());
	let mut resumed = TradingEnvironment::new(&engine, c, &[source.clone()])?;
	resumed.restore(&decoded)?;
	for env in [&mut first, &mut resumed] {
		// A newer flat suggestion cannot retroactively cancel an already pending IOC.
		env.step(&Matrix::new_from_slice(&engine, [1], &[1_i32])?)?;
		let event = env.submit()?;
		env.wait(&event)?;
		env.check_status()?;
		let ledger = env.ledger().read_f32()?;
		assert_eq!(ledger[1], 10.0);
		assert_eq!(ledger[7], 1.0);
		assert_eq!(ledger[12], 0.0);
	}
	assert_eq!(first.ledger().read_f32()?, resumed.ledger().read_f32()?);
	assert_eq!(
		first.observation().read_f32()?,
		resumed.observation().read_f32()?
	);
	source.market[0][0] = 101.0;
	let mut changed = TradingEnvironment::new(&engine, c, &[source])?;
	assert!(changed.restore(&decoded).is_err());
	first.close()?;
	resumed.close()?;
	changed.close()?;
	Ok(())
});

test_vk!(trading_mark_margin_liquidation_and_commission, engine, {
	use oa::ml::environment::TradingMarginTier;
	let c = event_config(5);
	let mut source = event_series(c.horizon);
	source.execution.leverage = 20.0;
	source.execution.quantity_step = 1.0;
	source.execution.liquidation_fee = 0.005;
	source.execution.margin_tiers = vec![TradingMarginTier {
		notional_limit: 1_000_000.0,
		initial_margin_rate: 0.05,
		maintenance_margin_rate: 0.025,
		maintenance_deduction: 0.0,
	}];
	source.market[3] = [98.0, 98.0, 0.0];
	source.market[4] = [97.4, 97.4, 0.0];
	let mut env = TradingEnvironment::new(&engine, c, &[source])?;
	env.reset(c.seed)?;
	let event = env.submit()?;
	env.wait(&event)?;
	for i in 0..3 {
		let step = env.step(&Matrix::new_from_slice(&engine, [1], &[2_i32])?)?;
		let event = env.submit()?;
		env.wait(&event)?;
		env.check_status()?;
		let ledger = env.ledger().read_f32()?;
		if i == 0 {
			assert_eq!(ledger[1], 200.0);
			assert_eq!(ledger[18], 1000.0);
			assert_eq!(ledger[19], 500.0);
		}
		if i == 1 {
			assert!((ledger[0] - 600.0).abs() < 0.001);
			assert_eq!(step.terminated().read::<u8>()?, [0]);
		}
		if i == 2 {
			assert_eq!(ledger[1], 0.0);
			assert_eq!(ledger[16], 1.0);
			assert!((ledger[0] - 382.6).abs() < 0.005);
			assert!((ledger[4] - 97.4).abs() < 0.001);
			assert_eq!(step.terminated().read::<u8>()?, [1]);
		}
	}
	env.close()?;
	Ok(())
});

test_vk!(trading_required_close_cannot_invent_liquidity, engine, {
	use oa::ml::environment::TradingBook;
	let c = event_config(1);
	let mut source = event_series(c.horizon);
	source.execution.quantity_step = 1.0;
	source.books = vec![
		TradingBook {
			bids: vec![[99.0, 1.0]],
			asks: vec![[101.0, 5.0]],
			mark_price: 100.0
		};
		3
	];
	let mut env = TradingEnvironment::new(&engine, c, &[source])?;
	env.reset(c.seed)?;
	let event = env.submit()?;
	env.wait(&event)?;
	env.step(&Matrix::new_from_slice(&engine, [1], &[2_i32])?)?;
	let event = env.submit()?;
	env.wait(&event)?;
	assert!(env.check_status().is_err());
	let ledger = env.ledger().read_f32()?;
	assert_eq!(ledger[0], 1000.0);
	assert_eq!(ledger[1], 0.0);
	assert_eq!(ledger[7], 0.0);
	env.close()?;
	Ok(())
});

// Decimal oracle: 0.012 opened, 0.011 partially closed, then 0.001 closed.
// FP32 subtraction used to leave 0.0009999992, which a second floor erased.
test_vk!(trading_fractional_lot_partial_close_reaches_flat, engine, {
	use oa::ml::environment::TradingBook;
	for action in [0_i32, 2] {
		let mut c = event_config(3);
		c.capital = 1.25;
		c.taker_fee = 0.001;
		let mut source = event_series(c.horizon);
		source.execution.quantity_step = 0.001;
		source.books = vec![
			TradingBook {
				bids: vec![[99.0, 1.0]],
				asks: vec![[101.0, 1.0]],
				mark_price: 100.0,
			};
			5
		];
		// Eleven whole lots fit; the half-lot of excess capacity must not fill.
		source.books[3].bids[0][1] = 0.0115;
		source.books[3].asks[0][1] = 0.0115;
		let mut env = TradingEnvironment::new(&engine, c, std::slice::from_ref(&source))?;
		env.reset(c.seed)?;
		let event = env.submit()?;
		env.wait(&event)?;
		for (index, selected) in [action, 1, 1].into_iter().enumerate() {
			env.step(&Matrix::new_from_slice(&engine, [1], &[selected])?)?;
			let event = env.submit()?;
			env.wait(&event)?;
			env.check_status()?;
			let state = env.ledger().read_f32()?;
			if index == 0 {
				assert!((state[1].abs() - 0.012).abs() < 1e-8, "opened {}", state[1]);
			} else if index == 1 {
				assert!(
					(state[1].abs() - 0.001).abs() < 1e-8,
					"remaining {}",
					state[1]
				);
				let snapshot =
					oa::ml::environment::TradingState::new_from_bytes(&env.snapshot()?.to_bytes())?;
				let mut restored = TradingEnvironment::new(&engine, c, std::slice::from_ref(&source))?;
				restored.restore(&snapshot)?;
				env.close()?;
				env = restored;
			} else {
				assert_eq!(state[1], 0.0);
				assert_eq!(state[7], 3.0);
				assert!((state[0] - 1.2236).abs() < 1e-6);
				assert!((state[4] - 0.0024).abs() < 1e-7);
				assert!((state[6] - 0.024).abs() < 1e-6);
			}
		}
		env.close()?;
	}
	Ok(())
});
