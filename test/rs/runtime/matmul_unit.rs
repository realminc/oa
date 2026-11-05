//! Host-side matmul planner legality, identity, and rejection evidence tests.

use super::{
	CAP_COOP_MAT1_BF16_INPUT, CAP_NAIVE_F32, CAP_SMALLM_F32, CAP_TILED_F32, GemmEpilogue, GemmKernel,
	GemmPrecision, INVALID_MATMUL_VARIANT_ID, KernelSelectionKind, MatmulDeviceContract,
	MatmulPreference, MatmulProblem, MatmulRejection, MatmulRouteCache, MatmulRouteValue,
	StoragePrecision, VARIANTS, caps_satisfy, classify_selection, device_contract_hash, find_variant,
	matmul_variant_id_from_name, plan, problem_contract_hash, registry_build_id, route_key,
	validate_plan, variant_legal,
};
use crate::runtime::{BufferAccess, KernelId, PushConstant};

fn contract() -> MatmulDeviceContract {
	MatmulDeviceContract {
		vendor_id: 0x8086,
		device_id: 0x0123,
		driver_id: 7,
		driver_version: 0x00e0_0000,
		capability_identity: 0x9e37_79b9_7f4a_7c15,
		caps_mask: CAP_TILED_F32 | CAP_NAIVE_F32 | CAP_SMALLM_F32,
		max_compute_workgroup_invocations: 1024,
		max_compute_workgroup_size: [1024, 1024, 64],
		max_compute_shared_memory_bytes: 256 * 1024,
	}
}

fn tiled() -> &'static super::MatmulVariant {
	find_variant(matmul_variant_id_from_name("GemmTiled")).expect("GemmTiled must be registered")
}

fn naive() -> &'static super::MatmulVariant {
	find_variant(matmul_variant_id_from_name("GemmNaive")).expect("GemmNaive must be registered")
}

fn small_m() -> &'static super::MatmulVariant {
	find_variant(matmul_variant_id_from_name("GemmSmallM")).expect("GemmSmallM must be registered")
}

fn measured(winner: u64) -> MatmulRouteValue {
	MatmulRouteValue {
		winner_variant: winner,
		median_gpu_time_ms: 1.0,
		p95_gpu_time_ms: 1.5,
		sample_count: 8,
		last_updated_step: 42,
	}
}

#[test]
fn registry_pins_exact_kernel_artifact_identity() {
	let variant = tiled();
	assert_eq!(variant.kernel_id, KernelId::MatrixMatMulNtTiledF32);
	assert_eq!(variant.kernel, GemmKernel::TiledF32);
	assert_eq!(variant.tile_m(), 64);
	assert_eq!(variant.tile_n(), 64);
	assert_eq!(variant.tile_k, 16);
	assert_eq!(variant.workgroup_invocations(), 256);
	assert!(variant.shared_memory_bytes().is_ok());
	assert_ne!(variant.shader_content_id(), 0);
	assert_ne!(registry_build_id(), 0);
}

#[test]
fn naive_candidate_pins_exact_artifact_identity() {
	let variant = naive();
	assert_eq!(variant.kernel_id, KernelId::MatrixMatMulNtNaiveF32);
	assert_eq!(variant.kernel, GemmKernel::Naive);
	assert_eq!(variant.tile_m(), 1);
	assert_eq!(variant.tile_n(), 1);
	assert_eq!(variant.tile_k, 1);
	assert_eq!(variant.workgroup_invocations(), 256);
	assert_eq!(variant.required_caps_mask, CAP_NAIVE_F32);
	assert_eq!(variant.shared_memory_bytes().unwrap_or(0), 0);
}

#[test]
fn gemv_candidate_pins_exact_artifact_identity() {
	let variant = small_m();
	assert_eq!(variant.kernel_id, KernelId::MatrixMatMulNtGemvF32);
	assert_eq!(variant.kernel, GemmKernel::SmallMF32);
	assert_eq!(variant.tile_m(), 1);
	assert_eq!(variant.tile_n(), 1);
	assert_eq!(variant.tile_k, 1);
	assert_eq!(variant.workgroup_invocations(), 256);
	assert_eq!(variant.required_caps_mask, CAP_SMALLM_F32);
	assert_eq!(variant.max_m, 4);
	assert!(variant.shared_memory_bytes().unwrap_or(0) > 0);
}

#[test]
fn small_m_problem_prefers_the_gemv_candidate() {
	let contract = contract();
	let problem = MatmulProblem::contiguous(1, 512, 512);
	let planned = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(planned.variant, matmul_variant_id_from_name("GemmSmallM"));
	assert_eq!(planned.kernel_id, KernelId::MatrixMatMulNtGemvF32);
	assert_eq!(planned.grid, [1, 512, 1]);
	assert_eq!(planned.selection, KernelSelectionKind::Direct);
	assert!(validate_plan(&contract, &planned, &problem));
	// Structural preference is bounded to the measured decode region
	// (m == 1, n >= 512, k >= 512); m <= 4 stays candidacy-legal so
	// required-variant pins and measured winners can still promote the kernel
	// where the tuner proves it faster.
	for (m, n, k) in [
		(1_u32, 256_u32, 512_u32),
		(1, 512, 256),
		(2, 512, 512),
		(4, 512, 512),
	] {
		let problem = MatmulProblem::contiguous(m, n, k);
		assert_eq!(
			variant_legal(&contract, small_m(), &problem, GemmPrecision::Auto),
			None,
			"m={m} n={n} k={k} must stay candidacy-legal"
		);
		let routed = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
		assert_eq!(
			routed.kernel_id,
			KernelId::MatrixMatMulNtTiledF32,
			"m={m} n={n} k={k}"
		);
	}
}

#[test]
fn untuned_tiny_decode_keeps_tiled_but_measured_small_m_remains_legal() {
	let contract = contract();
	let problem = MatmulProblem::contiguous(1, 256, 256);
	let preference = MatmulPreference::default();
	assert_eq!(
		plan(&contract, &problem, &preference, None).unwrap().kernel,
		GemmKernel::TiledF32
	);
	assert_eq!(
		plan(&contract, &problem, &preference, Some(small_m().id))
			.unwrap()
			.kernel,
		GemmKernel::SmallMF32
	);
}

#[test]
fn small_m_bound_falls_through_to_tiled() {
	let contract = contract();
	let variant = small_m();
	let problem = MatmulProblem::contiguous(5, 128, 64);
	assert_eq!(
		variant_legal(&contract, variant, &problem, GemmPrecision::Auto),
		Some(MatmulRejection::MaxMExceeded)
	);
	let plan = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtTiledF32);
}

#[test]
fn missing_small_m_capability_falls_back_to_tiled() {
	let mut contract = contract();
	contract.caps_mask &= !CAP_SMALLM_F32;
	let problem = MatmulProblem::contiguous(1, 512, 512);
	let plan = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtTiledF32);
}

#[test]
fn tiny_problem_prefers_the_naive_candidate() {
	let contract = contract();
	let problem = MatmulProblem::contiguous(4, 4, 16);
	let plan = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtNaiveF32);
	assert_eq!(plan.selection, KernelSelectionKind::NaiveFallback);
	assert_eq!(plan.grid, [1, 1, 1]);
}

#[test]
fn missing_tiled_capability_falls_back_to_naive() {
	let mut contract = contract();
	contract.caps_mask = CAP_NAIVE_F32;
	let problem = MatmulProblem::contiguous(128, 256, 64);
	let plan = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtNaiveF32);
	assert_eq!(plan.selection, KernelSelectionKind::NaiveFallback);
	assert_eq!(plan.grid, [128, 1, 1]);
}

#[test]
fn required_variant_pins_naive_for_diagnostics() {
	let contract = contract();
	let preference = MatmulPreference {
		required_variant: matmul_variant_id_from_name("GemmNaive"),
		max_workspace_bytes: 0,
		use_measured_cache: false,
	};
	let problem = MatmulProblem::contiguous(128, 256, 64);
	let plan = plan(&contract, &problem, &preference, None).unwrap();
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtNaiveF32);
}

#[test]
fn canonical_problem_is_legal_and_plans_direct_tiled() {
	let problem = MatmulProblem::contiguous(128, 256, 64);
	assert!(problem.is_canonical());
	assert_eq!(
		variant_legal(&contract(), tiled(), &problem, GemmPrecision::Auto),
		None
	);
	let plan = plan(&contract(), &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.variant, matmul_variant_id_from_name("GemmTiled"));
	assert_eq!(plan.kernel_id, KernelId::MatrixMatMulNtTiledF32);
	assert_eq!(plan.grid, [2, 4, 1]);
	assert_eq!(plan.workspace_bytes, 0);
	assert_eq!(plan.split_k, 1);
	assert_eq!(plan.selection, KernelSelectionKind::Direct);
	assert_eq!(plan.actual_precision, GemmPrecision::F32);
	assert_eq!(plan.reduction_kernel_id, None);
	assert_eq!(plan.problem_contract_hash, problem_contract_hash(&problem));
	assert_eq!(plan.device_contract_hash, device_contract_hash(&contract()));
	assert_eq!(plan.registry_build_id, registry_build_id());
	assert!(validate_plan(&contract(), &plan, &problem));
}

#[test]
fn odd_and_singleton_extents_divide_tiles_upward() {
	for (m, n, expected) in [
		(1_u32, 1_u32, [1, 1, 1]),
		(63, 65, [1, 2, 1]),
		(64, 64, [1, 1, 1]),
		(65, 128, [2, 2, 1]),
	] {
		let plan = plan(
			&contract(),
			&MatmulProblem::contiguous(m, n, 32),
			&MatmulPreference::default(),
			None,
		)
		.unwrap();
		assert_eq!(plan.grid, expected, "m={m} n={n}");
	}
}

#[test]
fn zero_extents_reject_every_candidate() {
	for (m, n, k) in [(0, 4, 4), (4, 0, 4), (4, 4, 0)] {
		let problem = MatmulProblem::contiguous(m, n, k);
		assert_eq!(
			variant_legal(&contract(), tiled(), &problem, GemmPrecision::Auto),
			Some(MatmulRejection::EmptyExtents)
		);
		assert!(plan(&contract(), &problem, &MatmulPreference::default(), None).is_err());
	}
}

#[test]
fn non_canonical_layout_is_rejected() {
	let mut problem = MatmulProblem::contiguous(64, 64, 64);
	problem.a.offset = 8;
	assert!(!problem.is_canonical());
	assert_eq!(
		variant_legal(&contract(), tiled(), &problem, GemmPrecision::Auto),
		Some(MatmulRejection::NonCanonicalLayout)
	);
	// A canonical-looking stride set with the contiguous flags cleared is also
	// a view contract, not a contiguous operand.
	let mut flagged = MatmulProblem::contiguous(64, 64, 64);
	flagged.a_contiguous = false;
	assert_eq!(
		variant_legal(&contract(), tiled(), &flagged, GemmPrecision::Auto),
		Some(MatmulRejection::NonCanonicalLayout)
	);
}

#[test]
fn batched_problem_is_rejected_by_single_dispatch_variants() {
	let mut problem = MatmulProblem::contiguous(64, 64, 64);
	problem.batch_count = 2;
	assert_eq!(
		variant_legal(&contract(), tiled(), &problem, GemmPrecision::Auto),
		Some(MatmulRejection::BatchUnsupported)
	);
	assert!(plan(&contract(), &problem, &MatmulPreference::default(), None).is_err());
}

#[test]
fn untransposed_b_is_rejected() {
	let mut problem = MatmulProblem::contiguous(64, 64, 64);
	problem.b_transposed = false;
	assert_eq!(
		variant_legal(&contract(), tiled(), &problem, GemmPrecision::Auto),
		Some(MatmulRejection::RequiresTransposedB)
	);
}

#[test]
fn epilogue_and_output_precision_contracts_are_checked() {
	let mut epilogue = MatmulProblem::contiguous(64, 64, 64);
	epilogue.epilogue = GemmEpilogue::Bias;
	assert_eq!(
		variant_legal(&contract(), tiled(), &epilogue, GemmPrecision::Auto),
		Some(MatmulRejection::EpilogueMismatch)
	);
	let mut output = MatmulProblem::contiguous(64, 64, 64);
	output.requested_output = StoragePrecision::Bf16;
	assert_eq!(
		variant_legal(&contract(), tiled(), &output, GemmPrecision::Auto),
		Some(MatmulRejection::OutputPrecisionMismatch)
	);
	let mut dual = MatmulProblem::contiguous(64, 64, 64);
	dual.requires_pre_activation = true;
	assert_eq!(
		variant_legal(&contract(), tiled(), &dual, GemmPrecision::Auto),
		Some(MatmulRejection::EpilogueMismatch)
	);
}

#[test]
fn missing_capability_rejects() {
	let mut limited = contract();
	limited.caps_mask &= !CAP_TILED_F32;
	assert_eq!(
		variant_legal(
			&limited,
			tiled(),
			&MatmulProblem::contiguous(4, 4, 4),
			GemmPrecision::Auto
		),
		Some(MatmulRejection::MissingDeviceCapability)
	);
	assert!(!caps_satisfy(limited.caps_mask, CAP_TILED_F32));
	assert!(caps_satisfy(limited.caps_mask, CAP_NAIVE_F32));
}

#[test]
fn device_limits_reject() {
	let problem = MatmulProblem::contiguous(64, 64, 64);
	for (field, expected) in [
		("invocations", MatmulRejection::WorkgroupInvocationLimit),
		("size", MatmulRejection::WorkgroupSizeLimit),
		("shared", MatmulRejection::SharedMemoryLimit),
	] {
		let mut limited = contract();
		match field {
			"invocations" => limited.max_compute_workgroup_invocations = 128,
			"size" => limited.max_compute_workgroup_size[0] = 128,
			_ => {
				limited.max_compute_shared_memory_bytes =
					tiled().shared_memory_bytes().unwrap().saturating_sub(1);
			}
		}
		assert_eq!(
			variant_legal(&limited, tiled(), &problem, GemmPrecision::Auto),
			Some(expected),
			"{field} limit"
		);
	}
}

#[test]
fn required_variant_isolation_succeeds_and_fails_closed() {
	let problem = MatmulProblem::contiguous(64, 64, 64);
	let pinned = MatmulPreference {
		required_variant: matmul_variant_id_from_name("GemmTiled"),
		max_workspace_bytes: 0,
		use_measured_cache: false,
	};
	assert_eq!(
		plan(&contract(), &problem, &pinned, None).unwrap().variant,
		matmul_variant_id_from_name("GemmTiled")
	);
	let unknown = MatmulPreference {
		required_variant: 0xdead_beef,
		..pinned
	};
	assert!(plan(&contract(), &problem, &unknown, None).is_err());
	let mut limited = contract();
	limited.caps_mask &= !CAP_TILED_F32;
	assert!(plan(&limited, &problem, &pinned, None).is_err());
}

#[test]
fn bf16_resolution_prefers_tensor_storage_inputs() {
	let mut problem = MatmulProblem::contiguous(64, 64, 64);
	problem.a_master = StoragePrecision::Bf16;
	assert_eq!(super::resolve_precision(&problem), GemmPrecision::Bf16);
	problem.precision_hint = GemmPrecision::F32;
	assert_eq!(super::resolve_precision(&problem), GemmPrecision::F32);
	// No BF16 candidate is registered: a BF16 request falls back to the F32
	// family and is classified as precision fallback evidence.
	let mut hinted = MatmulProblem::contiguous(64, 64, 64);
	hinted.precision_hint = GemmPrecision::Bf16;
	let plan = plan(&contract(), &hinted, &MatmulPreference::default(), None).unwrap();
	assert_eq!(plan.actual_precision, GemmPrecision::F32);
	assert_eq!(plan.selection, KernelSelectionKind::PrecisionFallback);
	assert_eq!(
		classify_selection(&plan, &hinted),
		KernelSelectionKind::PrecisionFallback
	);
}

#[test]
fn problem_contract_hash_covers_layout_and_flags() {
	let base = MatmulProblem::contiguous(8, 16, 32);
	let base_hash = problem_contract_hash(&base);
	assert_eq!(problem_contract_hash(&base), base_hash);
	let mut different = MatmulProblem::contiguous(8, 16, 32);
	different.training = false;
	assert_ne!(problem_contract_hash(&different), base_hash);
	different = MatmulProblem::contiguous(8, 16, 32);
	different.b.batch_stride = different.b.batch_stride.saturating_add(1);
	assert_ne!(problem_contract_hash(&different), base_hash);
	different = MatmulProblem::contiguous(8, 16, 33);
	assert_ne!(problem_contract_hash(&different), base_hash);
}

#[test]
fn device_contract_hash_covers_identity_and_limits() {
	let base = contract();
	let base_hash = device_contract_hash(&base);
	for mutate in [
		|c: &mut MatmulDeviceContract| c.vendor_id += 1,
		|c: &mut MatmulDeviceContract| c.device_id += 1,
		|c: &mut MatmulDeviceContract| c.driver_id += 1,
		|c: &mut MatmulDeviceContract| c.driver_version += 1,
		|c: &mut MatmulDeviceContract| c.capability_identity += 1,
		|c: &mut MatmulDeviceContract| c.caps_mask |= CAP_COOP_MAT1_BF16_INPUT,
		|c: &mut MatmulDeviceContract| c.max_compute_workgroup_invocations += 1,
		|c: &mut MatmulDeviceContract| c.max_compute_workgroup_size[1] += 1,
		|c: &mut MatmulDeviceContract| c.max_compute_shared_memory_bytes += 1,
	] {
		let mut changed = base;
		mutate(&mut changed);
		assert_ne!(device_contract_hash(&changed), base_hash);
	}
}

#[test]
fn validate_plan_rejects_stale_and_tampered_records() {
	let contract = contract();
	let problem = MatmulProblem::contiguous(64, 64, 64);
	let plan = plan(&contract, &problem, &MatmulPreference::default(), None).unwrap();
	assert!(validate_plan(&contract, &plan, &problem));
	type Tamper = fn(&mut super::MatmulPlan);
	let tampers: [(&str, Tamper); 9] = [
		("variant", |p| p.variant = INVALID_MATMUL_VARIANT_ID),
		("kernel", |p| p.kernel = GemmKernel::Naive),
		("grid", |p| p.grid[0] += 1),
		("workspace", |p| p.workspace_bytes = 4096),
		("problem hash", |p| p.problem_contract_hash ^= 1),
		("device hash", |p| p.device_contract_hash ^= 1),
		("registry", |p| p.registry_build_id ^= 1),
		("shader", |p| p.shader_content_hash = 0),
		("split_k", |p| p.split_k = 4),
	];
	for (label, tamper) in tampers {
		let mut stale = plan;
		tamper(&mut stale);
		assert!(
			!validate_plan(&contract, &stale, &problem),
			"{label} tamper must fail"
		);
	}
	// A plan is valid only for the exact problem it was planned for.
	assert!(!validate_plan(
		&contract,
		&plan,
		&MatmulProblem::contiguous(64, 64, 65)
	));
	assert!(!validate_plan(
		&{
			let mut other = contract;
			other.vendor_id += 1;
			other
		},
		&plan,
		&problem
	));
}

#[test]
fn every_registered_variant_reports_rejection_or_acceptance() {
	assert!(!VARIANTS.is_empty());
	for variant in VARIANTS {
		let m = if variant.max_m == 0 {
			64
		} else {
			variant.max_m
		};
		let mut problem = MatmulProblem::contiguous(m, 64, 64);
		problem.epilogue = variant.epilogue;
		let outcome = variant_legal(&contract(), variant, &problem, GemmPrecision::Auto);
		assert!(outcome.is_none(), "{} unexpectedly rejected", variant.name);
	}
}

#[test]
fn measured_winner_replays_only_when_opted_in_and_legal() {
	// Canonical shape: structural policy selects tiled.
	let problem = MatmulProblem::contiguous(128, 256, 64);
	let structural = plan(&contract(), &problem, &MatmulPreference::default(), None).unwrap();
	assert_eq!(structural.kernel_id, KernelId::MatrixMatMulNtTiledF32);

	// A legal measured winner overrides structural selection.
	let naive_id = matmul_variant_id_from_name("GemmNaive");
	let replayed = plan(
		&contract(),
		&problem,
		&MatmulPreference::default(),
		Some(naive_id),
	)
	.unwrap();
	assert_eq!(replayed.variant, naive_id);
	assert_eq!(replayed.selection, KernelSelectionKind::NaiveFallback);
	assert!(validate_plan(&contract(), &replayed, &problem));

	// Opt-out falls back to the structural route even with a winner recorded.
	let disabled = MatmulPreference {
		use_measured_cache: false,
		..MatmulPreference::default()
	};
	let opted_out = plan(&contract(), &problem, &disabled, Some(naive_id)).unwrap();
	assert_eq!(opted_out.kernel_id, KernelId::MatrixMatMulNtTiledF32);
}

#[test]
fn measured_winner_is_revalidated_not_trusted() {
	// Tiny shape: structural policy selects naive; a recorded tiled winner
	// must replay only because tiled is legal here.
	let problem = MatmulProblem::contiguous(4, 8, 16);
	let tiled_id = matmul_variant_id_from_name("GemmTiled");
	let replayed = plan(
		&contract(),
		&problem,
		&MatmulPreference::default(),
		Some(tiled_id),
	)
	.unwrap();
	assert_eq!(replayed.variant, tiled_id);

	// An unregistered winner falls through to the structural route.
	let unknown = plan(
		&contract(),
		&problem,
		&MatmulPreference::default(),
		Some(0xdead_beef),
	)
	.unwrap();
	assert_eq!(unknown.variant, matmul_variant_id_from_name("GemmNaive"));

	// A winner whose capability is absent on this device never replays.
	let mut limited = contract();
	limited.caps_mask &= !CAP_TILED_F32;
	let missing = plan(
		&limited,
		&problem,
		&MatmulPreference::default(),
		Some(tiled_id),
	)
	.unwrap();
	assert_eq!(missing.variant, matmul_variant_id_from_name("GemmNaive"));
}

#[test]
fn required_variant_wins_over_measured_cache() {
	let problem = MatmulProblem::contiguous(128, 256, 64);
	let pinned = MatmulPreference {
		required_variant: matmul_variant_id_from_name("GemmTiled"),
		use_measured_cache: false,
		max_workspace_bytes: 0,
	};
	let plan = plan(
		&contract(),
		&problem,
		&pinned,
		Some(matmul_variant_id_from_name("GemmNaive")),
	)
	.unwrap();
	assert_eq!(plan.variant, matmul_variant_id_from_name("GemmTiled"));
}

#[test]
fn route_key_covers_device_registry_and_complete_problem() {
	let problem = MatmulProblem::contiguous(64, 64, 64);
	let base = route_key(&contract(), &problem);

	let mut other_problem = problem;
	other_problem.k += 1;
	assert_ne!(base, route_key(&contract(), &other_problem));

	let mut other_contract = contract();
	other_contract.driver_version += 1;
	assert_ne!(base, route_key(&other_contract, &problem));

	// Driver/device identity is part of the key, but capability and limit
	// fields are not — those revalidate the winner at replay time instead.
	let mut weaker = contract();
	weaker.caps_mask &= !CAP_TILED_F32;
	weaker.max_compute_shared_memory_bytes /= 2;
	assert_eq!(base, route_key(&weaker, &problem));

	let value = measured(matmul_variant_id_from_name("GemmNaive"));
	assert_eq!(value.sample_count, 8);
}

#[test]
fn persisted_route_cache_round_trips_exact_records() {
	let mut cache = MatmulRouteCache::new();
	let key_a = route_key(&contract(), &MatmulProblem::contiguous(64, 64, 64));
	let key_b = route_key(&contract(), &MatmulProblem::contiguous(4, 8, 16));
	cache.publish(
		key_a,
		matmul_variant_id_from_name("GemmTiled"),
		1.25,
		1.5,
		12,
	);
	cache.publish(
		key_b,
		matmul_variant_id_from_name("GemmNaive"),
		0.01,
		0.02,
		30,
	);

	// Serialization is deterministic for identical cache content.
	let bytes = cache.serialize();
	assert_eq!(bytes, cache.serialize());

	let loaded = MatmulRouteCache::deserialize(&bytes).expect("round trip must succeed");
	assert_eq!(
		loaded.query(&key_a),
		Some(matmul_variant_id_from_name("GemmTiled"))
	);
	assert_eq!(
		loaded.query(&key_b),
		Some(matmul_variant_id_from_name("GemmNaive"))
	);
	assert_eq!(loaded.publication_step, 2);
	let record = loaded.records.get(&key_a).expect("record survives");
	assert_eq!(record.median_gpu_time_ms, 1.25);
	assert_eq!(record.p95_gpu_time_ms, 1.5);
	assert_eq!(record.sample_count, 12);
	assert_eq!(record.last_updated_step, 1);
}

#[test]
fn persisted_route_cache_fails_closed_on_corruption() {
	let mut cache = MatmulRouteCache::new();
	let key = route_key(&contract(), &MatmulProblem::contiguous(64, 64, 64));
	cache.publish(key, matmul_variant_id_from_name("GemmTiled"), 1.25, 1.5, 12);
	let valid = cache.serialize();
	// Header: magic u64 @0, version u32 @8, count u64 @12. One record follows.
	const HEADER: usize = 20;
	const RECORD: usize = 106 + 28;
	assert_eq!(valid.len(), HEADER + RECORD);

	let reject = |mut bytes: Vec<u8>, label: &str| {
		assert!(
			MatmulRouteCache::deserialize(&bytes).is_err(),
			"{label} must fail closed"
		);
		bytes.clear();
	};

	let mut bad_magic = valid.clone();
	bad_magic[0] ^= 0xff;
	reject(bad_magic, "bad magic");

	let mut bad_version = valid.clone();
	bad_version[8] ^= 0xff;
	reject(bad_version, "bad version");

	let mut huge_count = valid.clone();
	huge_count[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
	reject(huge_count, "uncapped record count");

	reject(valid[..HEADER + 10].to_vec(), "truncated record");

	let mut trailing = valid.clone();
	trailing.push(0);
	reject(trailing, "trailing bytes");

	// Layout offsets inside the 106-byte key: vendor(4) device(8) driver(4)
	// driverVersion(8) registry(8) = 32; m@32, a.row@40, b.row@56, c.row@72;
	// enums start at 96 (a_master), bools at 101 (a_contiguous).
	for (offset, value, label) in [
		(HEADER + 96, 1_u8, "retired storage-precision enum"),
		(HEADER + 96, 7_u8, "out-of-range storage-precision enum"),
		(HEADER + 99, 9_u8, "out-of-range precision-hint enum"),
		(HEADER + 100, 6_u8, "out-of-range epilogue enum"),
		(HEADER + 101, 2_u8, "out-of-range boolean"),
		(HEADER + 105, 2_u8, "out-of-range trailing boolean"),
	] {
		let mut bytes = valid.clone();
		bytes[offset] = value;
		reject(bytes, label);
	}

	// Zero a dimension (m) and a row stride (a.row) at u32 offsets.
	for (offset, label) in [
		(HEADER + 32, "zero m"),
		(HEADER + 40, "zero a row stride"),
		(HEADER + 56, "zero b row stride"),
		(HEADER + 72, "zero c row stride"),
	] {
		let mut bytes = valid.clone();
		bytes[offset..offset + 4].copy_from_slice(&0_u32.to_le_bytes());
		reject(bytes, label);
	}

	// Value section starts at HEADER + 106: winner u64, median f32, p95 f32,
	// samples u32, step u64.
	const VALUE: usize = HEADER + 106;
	let mut bad_winner = valid.clone();
	bad_winner[VALUE..VALUE + 8].copy_from_slice(&0_u64.to_le_bytes());
	reject(bad_winner, "invalid winner identity");

	let mut nan_median = valid.clone();
	nan_median[VALUE + 8..VALUE + 12].copy_from_slice(&f32::NAN.to_le_bytes());
	reject(nan_median, "non-finite median");

	let mut negative_p95 = valid.clone();
	negative_p95[VALUE + 12..VALUE + 16].copy_from_slice(&(-1.0_f32).to_le_bytes());
	reject(negative_p95, "negative p95");

	let mut zero_samples = valid.clone();
	zero_samples[VALUE + 16..VALUE + 20].copy_from_slice(&0_u32.to_le_bytes());
	reject(zero_samples, "zero sample count");

	// Duplicate keys: repeat the same record and claim two records.
	let mut duplicate = valid.clone();
	duplicate[12..20].copy_from_slice(&2_u64.to_le_bytes());
	duplicate.extend_from_slice(&valid[HEADER..]);
	reject(duplicate, "duplicate route key");
}

#[test]
fn lowering_produces_the_standard_abi_for_registered_candidates() {
	let problem = MatmulProblem::contiguous(128, 256, 64);
	let planned = plan(&contract(), &problem, &MatmulPreference::default(), None).unwrap();
	let launch = super::lower_mat_mul_nt_plan(&planned, &problem).unwrap();
	assert_eq!(launch.kernel_id, KernelId::MatrixMatMulNtTiledF32);
	assert_eq!(launch.grid, planned.grid);
	assert_eq!(launch.buffer_count, 3);
	assert_eq!(&launch.buffer_order[..3], &[0, 1, 2]);
	assert_eq!(launch.buffer_accesses[0], BufferAccess::Read);
	assert_eq!(launch.buffer_accesses[1], BufferAccess::Read);
	assert_eq!(launch.buffer_accesses[2], BufferAccess::Write);
	assert_eq!(launch.push_count, 3);
	assert_eq!(
		&launch.push_constants[..launch.push_count],
		&[
			PushConstant::U32(problem.m),
			PushConstant::U32(problem.n),
			PushConstant::U32(problem.k)
		]
	);
}

#[test]
fn lowering_rejects_empty_plans_and_unlowered_kernels() {
	let problem = MatmulProblem::contiguous(64, 64, 64);
	let mut plan = plan(&contract(), &problem, &MatmulPreference::default(), None).unwrap();
	plan.variant = INVALID_MATMUL_VARIANT_ID;
	assert!(super::lower_mat_mul_nt_plan(&plan, &problem).is_err());

	plan.variant = matmul_variant_id_from_name("GemmTiled");
	plan.kernel = GemmKernel::StridedF32;
	assert!(super::lower_mat_mul_nt_plan(&plan, &problem).is_err());
}

#[test]
#[ignore = "requires a Vulkan device; measured route publication proof"]
fn tuner_benchmark_shape_qualifies_and_publishes_the_winner() {
	let Ok(engine) = crate::Engine::new() else {
		return;
	};

	// A sub-64 product shape exercises the full candidate competition: the
	// naive, small-M, and tiled variants are all collected (the donor's sweep
	// skips naive only at m*n >= 64), so the measured ranking and oracle gate
	// both run.
	let shape = super::gemm_tuner::GemmTunerShape {
		m: 4,
		n: 8,
		k: 8,
		name: "test_tiny",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	};
	let result = super::gemm_tuner::benchmark_shape(&engine, &shape, 2, 8).unwrap();
	assert_eq!(result.ranked_candidates.len(), 3);
	for candidate in &result.ranked_candidates {
		assert!(!candidate.name.is_empty());
		assert!(matches!(
			candidate.kernel,
			GemmKernel::TiledF32 | GemmKernel::Naive | GemmKernel::SmallMF32
		));
		assert!(candidate.median_time_ms.is_finite() && candidate.median_time_ms > 0.0);
		assert!(candidate.p95_time_ms >= candidate.median_time_ms);
		assert_eq!(candidate.sample_count, 4);
	}
	assert_eq!(result.ranked_candidates[0].variant, result.best_variant);
	assert!(result.best_time_ms.is_finite() && result.best_time_ms > 0.0);
	assert!(result.best_gflops.is_finite());
	assert_eq!(result.best_kernel, result.ranked_candidates[0].kernel);

	// A subsequent plan through the engine must replay the measured winner
	// for the exact tuned problem.
	let replayed =
		crate::runtime::plan_mat_mul_nt(&engine.handle(), shape.m, shape.n, shape.k).unwrap();
	assert_eq!(replayed.variant, result.best_variant);
}

#[test]
#[ignore = "requires a Vulkan device; runs the complete default shape suite"]
fn tuner_run_sweeps_the_default_shapes() {
	let Ok(engine) = crate::Engine::new() else {
		return;
	};
	let results = super::gemm_tuner::run(&engine, 3, 10).unwrap();
	// Every qualified shape publishes a finite winner timing; epilogue
	// shapes have no registered candidates yet and report FAILED.
	assert!(!results.is_empty());
	for result in &results {
		assert!(result.best_time_ms.is_finite() && result.best_time_ms > 0.0);
	}
}
