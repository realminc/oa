//! Single-use host entropy for the private GPU secret boundary.
//!
//! The OA C++ donor audit admits no GPU secret
//! execution or owned entropy staging. This Rust ownership boundary is new;
//! it does not replace a proven donor algorithm or admit a GPU operation.

use memmap2::MmapMut;

use crate::{Error, Result, core::memory::zero_secure};

/// Dedicated, page-locked host entropy. No Clone, Debug or byte accessor.
pub(crate) struct Entropy<const N: usize> {
	storage: MmapMut,
}

impl<const N: usize> Entropy<N> {
	pub(crate) fn generate() -> Result<Self> {
		Self::from_source(|bytes| {
			getrandom::fill(bytes)
				.map_err(|_| Error::internal("operating-system entropy generation failed"))
		})
	}

	fn from_source(source: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<Self> {
		if N != 32 && N != 64 {
			return Err(Error::invalid_argument(
				"PQC entropy must contain 32 or 64 bytes",
			));
		}
		#[cfg(not(unix))]
		{
			let _ = source;
			Err(Error::missing_capability(
				"locked host entropy requires a supported Unix adapter",
			))
		}
		#[cfg(unix)]
		{
			let storage = MmapMut::map_anon(N)
				.map_err(|e| Error::backend_failure("host entropy", "allocation", e))?;
			// A dedicated mapping avoids allocator pages shared with other secrets:
			// one owner's unmap cannot unlock another owner's range.
			storage
				.lock()
				.map_err(|e| Error::backend_failure("host entropy", "page locking", e))?;
			let mut entropy = Self { storage };
			// The owner exists before the source may partially write or panic.
			source(&mut entropy.storage)?;
			Ok(entropy)
		}
	}

	/// Consume once; the callback cannot return a borrow of the entropy.
	pub(crate) fn consume<T>(mut self, consumer: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
		consume_bytes(&mut self.storage, consumer)
	}
}

fn consume_bytes<T>(bytes: &mut [u8], consumer: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
	// Already locked by the owner; this borrowed guard adds mandatory erasure
	// across callback success, error and unwind. Owner Drop erases again before
	// the mapping is released, including generation failure and abandonment.
	let guard = EraseOnDrop(bytes);
	consumer(guard.0)
}

struct EraseOnDrop<'a>(&'a mut [u8]);

impl Drop for EraseOnDrop<'_> {
	fn drop(&mut self) {
		zero_secure(self.0);
	}
}

impl<const N: usize> Drop for Entropy<N> {
	fn drop(&mut self) {
		zero_secure(&mut self.storage);
		// MmapMut unmaps only after this erasure; unmap releases the page lock.
	}
}

#[cfg(test)]
#[path = "../../../test/rs/cryptography/entropy_unit.rs"]
mod tests;
