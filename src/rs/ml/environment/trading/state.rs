//! Explicit host checkpoint of environment state, never an execution owner.
use super::*;

/// Versioned complete analytical lane state, bound to source tables/config.
/// Snapshot only at exact completion; consumers persist this alongside model,
/// optimizer, PPO action cursor and their own source-receipt identity.
#[derive(Clone, Debug)]
pub struct TradingState {
	pub(super) config: u64,
	pub(super) source: crate::cryptography::Hash,
	pub(super) ledger: Vec<f32>,
	pub(super) metadata: Vec<u32>,
	pub(super) done: Vec<u8>,
	pub(super) observation: Vec<f32>,
}
impl TradingState {
	/// Encode exact FP32/U32 state with a versioned header and corruption digest.
	pub fn to_bytes(&self) -> Vec<u8> {
		let mut out = b"OATRAD03".to_vec();
		out.extend(self.config.to_le_bytes());
		out.extend(self.source.as_bytes());
		for n in [
			self.ledger.len(),
			self.metadata.len(),
			self.done.len(),
			self.observation.len(),
		] {
			out.extend((n as u64).to_le_bytes());
		}
		for n in &self.ledger {
			out.extend(n.to_bits().to_le_bytes());
		}
		for n in &self.metadata {
			out.extend(n.to_le_bytes());
		}
		out.extend(&self.done);
		for n in &self.observation {
			out.extend(n.to_bits().to_le_bytes());
		}
		let digest = crate::cryptography::hash(&out);
		out.extend(digest.as_bytes());
		out
	}
	/// # Errors
	/// Rejects unknown versions, malformed extents, nonfinite state or corruption.
	pub fn new_from_bytes(bytes: &[u8]) -> Result<Self> {
		let bad = || Error::invalid_argument("invalid trading environment checkpoint");
		if bytes.len() < 112 || bytes.len() > 256 * 1024 * 1024 || &bytes[..8] != b"OATRAD03" {
			return Err(bad());
		}
		let payload = &bytes[..bytes.len() - 32];
		if crate::cryptography::hash(payload).as_bytes() != &bytes[bytes.len() - 32..] {
			return Err(bad());
		}
		let word = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("checked header"));
		let mut sizes = [0; 4];
		for (i, n) in sizes.iter_mut().enumerate() {
			*n = usize::try_from(word(48 + i * 8)).map_err(|_| bad())?;
			if *n > 50_000_000 {
				return Err(bad());
			}
		}
		let expected = sizes[0]
			.checked_add(sizes[1])
			.and_then(|n| n.checked_add(sizes[3]))
			.and_then(|n| n.checked_mul(4))
			.and_then(|n| n.checked_add(sizes[2]))
			.and_then(|n| n.checked_add(80))
			.ok_or_else(bad)?;
		if expected != payload.len() {
			return Err(bad());
		}
		let mut at = 80;
		let mut take = |count: usize| {
			let slice = &payload[at..at + count];
			at += count;
			slice
		};
		let ledger = take(sizes[0] * 4)
			.as_chunks::<4>()
			.0
			.iter()
			.map(|v| f32::from_bits(u32::from_le_bytes(*v)))
			.collect::<Vec<_>>();
		let metadata = take(sizes[1] * 4)
			.as_chunks::<4>()
			.0
			.iter()
			.map(|v| u32::from_le_bytes(*v))
			.collect();
		let done = take(sizes[2]).to_vec();
		let observation = take(sizes[3] * 4)
			.as_chunks::<4>()
			.0
			.iter()
			.map(|v| f32::from_bits(u32::from_le_bytes(*v)))
			.collect::<Vec<_>>();
		if ledger.iter().chain(&observation).any(|v| !v.is_finite()) || done.iter().any(|v| *v > 1) {
			return Err(bad());
		}
		Ok(Self {
			config: word(8),
			source: crate::cryptography::Hash::new_from_bytes(&bytes[16..48])?,
			ledger,
			metadata,
			done,
			observation,
		})
	}
}

pub(super) fn fingerprint(
	tables: &TradingTables,
	series: &[TradingSeries],
) -> Result<crate::cryptography::Hash> {
	let mut hash = crate::cryptography::Hasher::new();
	hash.update(b"oa.trading.source.v3")?;
	for values in [&tables.market, &tables.features, &tables.rules] {
		hash.update(&(values.len() as u64).to_le_bytes())?;
		for chunk in values.chunks(4096) {
			let bytes = chunk
				.iter()
				.flat_map(|v| v.to_bits().to_le_bytes())
				.collect::<Vec<_>>();
			hash.update(&bytes)?;
		}
	}
	for v in &tables.bounds {
		hash.update(&v.to_le_bytes())?;
	}
	for s in series {
		for t in &s.timestamps {
			hash.update(&t.to_le_bytes())?;
		}
	}
	Ok(hash.finalize())
}
impl TradingEnvironment<'_> {
	/// Explicitly synchronize and read a portable exact environment snapshot.
	/// # Errors
	/// Requires healthy initialized state and no open environment transaction.
	pub fn snapshot(&self) -> Result<TradingState> {
		self.engine.checkpoint()?.wait()?;
		self.check_status()?;
		Ok(TradingState {
			config: self.config.identity(),
			source: self.source,
			ledger: self.ledger.read_f32()?,
			metadata: self.meta.read::<u32>()?,
			done: self.done.read::<u8>()?,
			observation: self.observation.read_f32()?,
		})
	}
	/// Restore into unchanged source/config and stable captured buffers.
	/// # Errors
	/// Rejects incompatible or unhealthy state before mutating any buffer.
	pub fn restore(&mut self, state: &TradingState) -> Result<()> {
		if self.execution.has_active_recording()
			|| self.execution.has_pending_event()
			|| state.config != self.config.identity()
			|| state.source != self.source
			|| state.ledger.len() != self.ledger.num_elements()
			|| state.metadata.len() != self.meta.num_elements()
			|| state.done.len() != self.done.num_elements()
			|| state.observation.len() != self.observation.num_elements()
		{
			return Err(Error::invalid_argument(
				"trading state/source/configuration mismatch",
			));
		}
		let bounds = self.bounds.read::<u32>()?;
		for (i, m) in state.metadata.as_chunks::<4>().0.iter().enumerate() {
			if m[0] < bounds[i * 2] || m[0] > bounds[i * 2 + 1] || m[1] > self.config.horizon || m[3] != 0
			{
				return Err(Error::invalid_argument(
					"invalid trading snapshot cursor/status",
				));
			}
		}
		self.engine.checkpoint()?.wait()?;
		self.committed = false;
		self.ledger.write_values(&state.ledger)?;
		self.meta.write_values(&state.metadata)?;
		self.done.write_values(&state.done)?;
		self.observation.write_values(&state.observation)?;
		self.committed = true;
		self.reset_pending = false;
		self.rollback_observation = None;
		Ok(())
	}
}
