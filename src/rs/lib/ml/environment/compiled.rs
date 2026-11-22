//! Compiled-environment contract (OA-CC-20).
//!
//! A compiled environment declares reset, step and observation programs that
//! `Engine::capture` records once and `CompiledRollout` replays across a fixed
//! PPO horizon without per-step host recording. The design record is
//! `oaCompiledEnvironment.md`.
//!
//! The seven-word rollout cursor shared by every replay program is:
//!
//! | Word | Role |
//! |---:|---|
//! | 0 | Step index, advanced on device |
//! | 1 | Philox/action-sampling state, low |
//! | 2 | Philox/action-sampling state, high |
//! | 3 | Reset-seed state, low |
//! | 4 | Reset-seed state, high |
//! | 5 | Final-step flag for terminal handling |
//! | 6 | Rollout append cursor |

use crate::{Matrix, Result};

use super::{Environment, EnvironmentTransition};

/// Replayable reset/step/observation contract for one [`Environment`].
///
/// Implementations must keep every replay-visible piece of episode state in
/// device-resident matrices: the captured plans bind those buffers once and
/// replay the identical command sequence, so no host-computed value may
/// differ across replays. Deterministic per-episode variation belongs in
/// on-device counters (as CartPole's `episode_index` does) or in the cursor
/// words above.
pub trait CompiledEnvironment: Environment {
	/// Whether the environment may record a replay plan at all.
	///
	/// Trading uses this to reject capture while its tables are stale;
	/// environments without externally supplied state return `true`.
	fn replay_valid(&self) -> bool {
		true
	}

	/// Whether committed device state lets a continuation collection skip the
	/// reset plan and capture a resume plan instead.
	fn replay_initialized(&self) -> bool;

	/// Environment-owned matrices whose initialized-state tracking the
	/// capture must mark: every buffer a replay program reads before writing.
	/// Write-only transition outputs are excluded.
	fn replay_slots(&self) -> Vec<&Matrix>;

	/// Stable identity over every input the captured plans depend on: state
	/// matrix value identities plus the complete recipe that changes command
	/// output. The per-collection seed is excluded — it arrives through
	/// cursor words 3 and 4.
	fn replay_identity(&self) -> Vec<u64>;

	/// Record the environment's reset program for plan capture.
	///
	/// `only_done` resets just the lanes whose `done` flag is set. `cursor`
	/// carries reset entropy in words 3 and 4 for environments that consume
	/// it; environments whose reset derives deterministically from on-device
	/// episode counters may ignore it.
	///
	/// # Errors
	///
	/// Returns the implementation's reset recording failures.
	fn record_reset_replay(&mut self, only_done: bool, cursor: &Matrix) -> Result<()>;

	/// Record the environment's step program for plan capture.
	///
	/// `action` is the policy-selected discrete/continuous action buffer for
	/// this step; `cursor` carries step entropy for environments that consume
	/// it. The returned transition must reference device matrices the replay
	/// appender can copy into cursor-addressed rollout storage.
	///
	/// # Errors
	///
	/// Returns the implementation's step recording failures.
	fn record_step_replay(
		&mut self,
		action: &Matrix,
		cursor: &Matrix,
	) -> Result<EnvironmentTransition>;

	/// Rebind the observation storage the captured reset/step plans write.
	///
	/// After plan capture the environment's observation must refer to the
	/// storage bound into the plans. Environments whose observation is a
	/// fixed state matrix (captured in place) implement this as a no-op.
	fn replay_observation(&mut self, observation: Matrix);

	/// Observe lane status after exact completion of a submitted chunk.
	///
	/// Environments with a device status word reject failed lanes here; the
	/// default accepts everything.
	///
	/// # Errors
	///
	/// Returns readback/completion failure or a rejected lane.
	fn check_status(&self) -> Result<()> {
		Ok(())
	}
}
