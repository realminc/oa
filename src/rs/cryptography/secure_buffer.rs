//! Borrowed host storage with best-effort page locking and mandatory erasure.

use crate::core::memory::zero_secure;

/// A non-owning mutable byte range that is securely erased when reset or
/// dropped.
///
/// Linux page locking is best effort and observable through [`Self::is_locked`].
/// The borrow guarantees that the backing range remains alive for this guard.
pub struct SecureBuffer<'a> {
	bytes: Option<&'a mut [u8]>,
	locked: bool,
}

impl<'a> SecureBuffer<'a> {
	/// Guard a mutable byte range and attempt to keep it out of swap.
	#[must_use]
	pub fn new(bytes: &'a mut [u8]) -> Self {
		#[cfg(target_os = "linux")]
		let locked = if bytes.is_empty() {
			false
		} else {
			// SAFETY: the mutable borrow proves that the range is live and uniquely
			// accessible for the complete lifetime of this guard.
			unsafe { libc::mlock(bytes.as_ptr().cast(), bytes.len()) == 0 }
		};
		#[cfg(not(target_os = "linux"))]
		let locked = false;

		Self {
			bytes: Some(bytes),
			locked,
		}
	}

	/// Return the guarded bytes.
	#[must_use]
	pub fn as_slice(&self) -> &[u8] {
		self.bytes.as_deref().unwrap_or_default()
	}

	/// Return the guarded bytes mutably.
	#[must_use]
	pub fn as_mut_slice(&mut self) -> &mut [u8] {
		self.bytes.as_deref_mut().unwrap_or_default()
	}

	/// Return the guarded range size in bytes.
	#[must_use]
	pub fn size_bytes(&self) -> usize {
		self.bytes.as_deref().map_or(0, <[u8]>::len)
	}

	/// Return whether this guard currently owns a non-empty range.
	#[must_use]
	pub fn is_valid(&self) -> bool {
		self.size_bytes() != 0
	}

	/// Return whether the operating system accepted the page-lock request.
	#[must_use]
	pub const fn is_locked(&self) -> bool {
		self.locked
	}

	/// Erase the guarded range without releasing it.
	pub fn secure_zero(&mut self) {
		if let Some(bytes) = self.bytes.as_deref_mut() {
			zero_secure(bytes);
		}
	}

	/// Erase and release the guarded range.
	pub fn reset(&mut self) {
		self.secure_zero();
		if let Some(bytes) = self.bytes.take() {
			#[cfg(target_os = "linux")]
			if self.locked {
				// SAFETY: `bytes` is the exact live range passed to the successful
				// `mlock` call and remains borrowed until after this call.
				unsafe {
					libc::munlock(bytes.as_ptr().cast(), bytes.len());
				}
			}
		}
		self.locked = false;
	}
}

impl Drop for SecureBuffer<'_> {
	fn drop(&mut self) {
		self.reset();
	}
}

impl std::fmt::Debug for SecureBuffer<'_> {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("SecureBuffer")
			.field("size_bytes", &self.size_bytes())
			.field("locked", &self.locked)
			.finish_non_exhaustive()
	}
}

#[cfg(test)]
mod tests {
	use super::SecureBuffer;

	#[test]
	fn reset_erases_the_borrowed_range() {
		let mut bytes = [0xa5; 32];
		{
			let mut guarded = SecureBuffer::new(&mut bytes);
			assert!(guarded.is_valid());
			assert_eq!(guarded.size_bytes(), 32);
			guarded.reset();
			assert!(!guarded.is_valid());
		}
		assert_eq!(bytes, [0; 32]);
	}

	#[test]
	fn drop_erases_the_borrowed_range() {
		let mut bytes = [0x5a; 17];
		{
			let _guarded = SecureBuffer::new(&mut bytes);
		}
		assert_eq!(bytes, [0; 17]);
	}
}
