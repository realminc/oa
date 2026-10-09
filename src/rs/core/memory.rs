//! Checked host-memory operations and explicit cache policy.
//!
//! Ordinary copies retain the platform's bulk-copy implementation while using
//! compiler-sized blocks for small dynamic lengths. Streaming copies are an
//! explicit contract for destinations that the CPU will not consume soon,
//! such as GPU upload storage. They never change copy correctness.

use std::sync::atomic::{Ordering, compiler_fence};

use super::{Error, Result};

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "memory/x86.rs"]
mod x86;

#[cfg(all(target_arch = "x86_64", not(miri)))]
const STREAMING_MIN_BYTES: usize = 1024;
#[cfg(all(target_arch = "x86_64", not(miri)))]
const STREAMING_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Copy one complete byte slice into another non-overlapping slice.
///
/// Small dynamic copies use fixed compiler-known blocks. Qualified medium
/// copies use cached SIMD stores; bulk copies retain the platform implementation.
///
/// # Errors
///
/// Returns an error when the slices have different lengths.
#[inline(always)]
pub fn copy(destination: &mut [u8], source: &[u8]) -> Result<()> {
	if destination.len() != source.len() {
		return Err(length_mismatch(
			destination.len(),
			source.len(),
			"memory::copy",
		));
	}
	// SAFETY: equal slice lengths prove both ranges contain `source.len()` live
	// bytes. Safe Rust's mutable/shared borrow rules make the ranges disjoint.
	unsafe {
		copy_to_ptr(destination.as_mut_ptr(), source.as_ptr(), source.len());
	}
	Ok(())
}

/// Copy one complete byte slice using an explicit one-way streaming policy.
///
/// The destination must not be consumed by the CPU soon. On a qualified x86-64
/// target, medium-sized copies use non-temporal stores. Other sizes use cached
/// stores, preferring the platform routine above the small-copy range.
///
/// # Errors
///
/// Returns an error when the slices have different lengths.
#[inline(always)]
pub fn copy_streaming(destination: &mut [u8], source: &[u8]) -> Result<()> {
	if destination.len() != source.len() {
		return Err(length_mismatch(
			destination.len(),
			source.len(),
			"memory::copy_streaming",
		));
	}
	// SAFETY: equal slice lengths prove both ranges contain `source.len()` live
	// bytes. Safe Rust's mutable/shared borrow rules make the ranges disjoint.
	unsafe {
		copy_streaming_to_ptr(destination.as_mut_ptr(), source.as_ptr(), source.len());
	}
	Ok(())
}

/// Compare two byte slices with ordinary early-exit semantics.
#[must_use]
pub fn equal(left: &[u8], right: &[u8]) -> bool {
	if left.len() != right.len() {
		return false;
	}
	if left.as_ptr() == right.as_ptr() || left.is_empty() {
		return true;
	}

	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if std::arch::is_x86_feature_detected!("avx512f")
		&& std::arch::is_x86_feature_detected!("avx512bw")
	{
		// SAFETY: the feature checks admit AVX-512F and AVX-512BW. Both
		// slices contain the same number of initialized bytes and remain live.
		return unsafe { x86::equal_avx512(left.as_ptr(), right.as_ptr(), left.len()) };
	}

	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if std::arch::is_x86_feature_detected!("avx2") {
		// SAFETY: the feature check admits AVX2. Both slices contain the same
		// number of initialized bytes and remain live for the call.
		return unsafe { x86::equal_avx2(left.as_ptr(), right.as_ptr(), left.len()) };
	}

	left == right
}

/// Compare equal-length byte slices without content-dependent early exit.
///
/// Different public lengths may return immediately. For equal lengths, every
/// byte is visited before the result is produced. This is a bounded primitive,
/// not a claim of system-wide constant-time behavior on every platform.
#[must_use]
pub fn equal_constant_time(left: &[u8], right: &[u8]) -> bool {
	if left.len() != right.len() {
		return false;
	}

	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if std::arch::is_x86_feature_detected!("avx512f")
		&& std::arch::is_x86_feature_detected!("avx512bw")
	{
		// SAFETY: the feature checks admit AVX-512F and AVX-512BW. Both
		// slices contain the same number of initialized bytes and remain live.
		return unsafe { x86::equal_constant_time_avx512(left.as_ptr(), right.as_ptr(), left.len()) };
	}

	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if std::arch::is_x86_feature_detected!("avx2") {
		// SAFETY: the feature check admits AVX2. Both slices contain the same
		// number of initialized bytes and remain live for the call.
		return unsafe { x86::equal_constant_time_avx2(left.as_ptr(), right.as_ptr(), left.len()) };
	}

	let mut difference = 0_u8;
	for (&left_byte, &right_byte) in left.iter().zip(right) {
		difference |= left_byte ^ right_byte;
	}
	compiler_fence(Ordering::SeqCst);
	difference == 0
}

/// Erase every byte through observable stores.
///
/// Unlike ordinary filling or zeroing, these writes are retained even when the
/// compiler can prove the destination is not subsequently read. This does not
/// erase copies held elsewhere or guarantee removal from caches or swap.
pub fn zero_secure(bytes: &mut [u8]) {
	for byte in bytes {
		// SAFETY: `byte` is a live, aligned mutable reference to one byte. The
		// volatile store is used solely to prevent dead-store elimination.
		unsafe {
			std::ptr::write_volatile(byte, 0);
		}
	}
	compiler_fence(Ordering::SeqCst);
}

#[cold]
#[inline(never)]
fn length_mismatch(destination: usize, source: usize, operation: &'static str) -> Error {
	Error::invalid_argument(format!(
		"{operation} requires equal lengths; destination is {destination} bytes, source is {source} bytes"
	))
}

/// Copy initialized bytes to a disjoint writable range.
///
/// # Safety
///
/// `destination` must be valid for writes of `length` bytes, `source` must be
/// valid for reads of `length` initialized bytes, and the two ranges must not
/// overlap. Both ranges must remain live for the call.
#[inline(always)]
pub(crate) unsafe fn copy_to_ptr(destination: *mut u8, source: *const u8, length: usize) {
	#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
	{
		// SAFETY: the caller provides disjoint readable/writable ranges.
		unsafe { std::ptr::copy_nonoverlapping(source, destination, length) };
	}
	#[cfg(not(all(target_arch = "x86_64", not(target_feature = "avx2"))))]
	// SAFETY: the caller proves the same range and aliasing contract.
	unsafe {
		copy_native_to_ptr(destination, source, length)
	};
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", not(target_feature = "avx2"))))]
unsafe fn copy_native_to_ptr(destination: *mut u8, source: *const u8, length: usize) {
	if length > 256 {
		#[cfg(all(target_arch = "x86_64", not(miri)))]
		if length <= 4096
			&& (cfg!(target_feature = "avx512f") || length >= 2048)
			&& std::arch::is_x86_feature_detected!("avx512f")
		{
			// SAFETY: AVX-512F is checked or guaranteed by this build target;
			// the caller proves live disjoint ranges and length is > 256.
			unsafe { x86::copy_cached_avx512(destination, source, length) };
			return;
		}
		// SAFETY: the caller proves disjoint readable/writable ranges. Give bulk
		// copies a direct path without walking the small-copy dispatch tree.
		unsafe {
			std::ptr::copy_nonoverlapping(source, destination, length);
		}
		return;
	}
	if length == 0 {
		return;
	}

	// SAFETY: the caller provides disjoint live ranges of `length` bytes. Every
	// fixed block below remains inside those ranges; blocks may overlap other
	// blocks from the same copy, but source and destination never overlap.
	unsafe {
		if length <= 16 {
			if length >= 8 {
				copy_block::<8>(destination, source);
				copy_block::<8>(destination.add(length - 8), source.add(length - 8));
			} else if length >= 4 {
				copy_block::<4>(destination, source);
				copy_block::<4>(destination.add(length - 4), source.add(length - 4));
			} else if length >= 2 {
				copy_block::<2>(destination, source);
				copy_block::<2>(destination.add(length - 2), source.add(length - 2));
			} else {
				copy_block::<1>(destination, source);
			}
			return;
		}

		if length <= 32 {
			copy_block::<16>(destination, source);
			copy_block::<16>(destination.add(length - 16), source.add(length - 16));
			return;
		}

		if length <= 64 {
			copy_block::<32>(destination, source);
			copy_block::<32>(destination.add(length - 32), source.add(length - 32));
			return;
		}

		if length <= 128 {
			copy_block::<32>(destination, source);
			copy_block::<32>(destination.add(32), source.add(32));
			copy_block::<32>(destination.add(length - 64), source.add(length - 64));
			copy_block::<32>(destination.add(length - 32), source.add(length - 32));
			return;
		}

		if length <= 256 {
			copy_block::<32>(destination, source);
			copy_block::<32>(destination.add(32), source.add(32));
			copy_block::<32>(destination.add(64), source.add(64));
			copy_block::<32>(destination.add(96), source.add(96));
			copy_block::<32>(destination.add(length - 128), source.add(length - 128));
			copy_block::<32>(destination.add(length - 96), source.add(length - 96));
			copy_block::<32>(destination.add(length - 64), source.add(length - 64));
			copy_block::<32>(destination.add(length - 32), source.add(length - 32));
		}
	}
}

/// Copy initialized bytes to a disjoint writable range using one-way policy.
///
/// # Safety
///
/// `destination` must be valid for writes of `length` bytes, `source` must be
/// valid for reads of `length` initialized bytes, and the two ranges must not
/// overlap. Both ranges must remain live for the call. The CPU must not consume
/// the destination soon after this function returns.
#[inline(always)]
pub(crate) unsafe fn copy_streaming_to_ptr(destination: *mut u8, source: *const u8, length: usize) {
	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if (STREAMING_MIN_BYTES..=STREAMING_MAX_BYTES).contains(&length)
		&& std::arch::is_x86_feature_detected!("avx512f")
		&& std::arch::is_x86_feature_detected!("avx512bw")
	{
		// SAFETY: the caller proves the pointer contract and the feature checks
		// admit AVX-512F and AVX-512BW. Arbitrary alignment and length are valid.
		unsafe {
			x86::copy_streaming_avx512(destination, source, length);
		}
		return;
	}

	#[cfg(all(target_arch = "x86_64", not(miri)))]
	if (STREAMING_MIN_BYTES..=STREAMING_MAX_BYTES).contains(&length)
		&& std::arch::is_x86_feature_detected!("avx2")
	{
		// SAFETY: the caller proves the pointer contract and the feature check
		// admits AVX2. The operation handles arbitrary byte alignment and length.
		unsafe {
			x86::copy_streaming_avx2(destination, source, length);
		}
		return;
	}

	// SAFETY: the caller proves disjoint live ranges. Outside the NT window,
	// one-way medium/bulk copies use the platform routine: the cached SIMD
	// path is tuned for reuse and can lose on a cold streaming arena.
	unsafe {
		if length > 256 {
			std::ptr::copy_nonoverlapping(source, destination, length);
		} else {
			copy_to_ptr(destination, source, length);
		}
	}
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", not(target_feature = "avx2"))))]
unsafe fn copy_block<const LENGTH: usize>(destination: *mut u8, source: *const u8) {
	// SAFETY: the caller proves both ranges contain `LENGTH` bytes and do not
	// overlap. The byte type imposes no additional alignment requirement.
	unsafe {
		std::ptr::copy_nonoverlapping(source, destination, LENGTH);
	}
}
