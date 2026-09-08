use std::arch::x86_64::{
	__m256i, __m512i, _mm_sfence, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
	_mm256_or_si256, _mm256_setzero_si256, _mm256_stream_si256, _mm256_xor_si256,
	_mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_or_si512, _mm512_setzero_si512,
	_mm512_stream_si512, _mm512_xor_si512,
};

// Native builds inline the cached path; portable builds call it only after
// runtime feature detection and above the size that amortizes that call.
#[cfg_attr(target_feature = "avx512f", inline(always))]
#[cfg_attr(not(target_feature = "avx512f"), target_feature(enable = "avx512f"))]
#[cfg_attr(not(target_feature = "avx512f"), inline)]
#[cfg(any(target_feature = "avx2", test))]
pub(super) unsafe fn copy_cached_avx512(destination: *mut u8, source: *const u8, length: usize) {
	use std::arch::x86_64::_mm512_storeu_si512;
	let mut offset = 0;
	let alignment = destination.addr().wrapping_neg() & 63;
	if alignment != 0 {
		// SAFETY: at least 256 bytes are available. Copying the first vector
		// covers the prefix before advancing to the next destination cache line.
		unsafe { _mm512_storeu_si512(destination.cast(), _mm512_loadu_si512(source.cast())) };
		offset = alignment;
	}
	while length - offset >= 256 {
		// SAFETY: at least 256 readable/writable bytes remain at offset.
		unsafe { copy_cached_block(destination.add(offset), source.add(offset)) };
		offset += 256;
	}
	while length - offset >= 64 {
		// SAFETY: this loop admits a complete vector within both ranges.
		unsafe {
			_mm512_storeu_si512(
				destination.add(offset).cast(),
				_mm512_loadu_si512(source.add(offset).cast()),
			);
		}
		offset += 64;
	}
	if offset != length {
		// SAFETY: the last 64 bytes lie inside both ranges. They can overlap
		// previously copied bytes, but source and destination remain disjoint.
		unsafe {
			_mm512_storeu_si512(
				destination.add(length - 64).cast(),
				_mm512_loadu_si512(source.add(length - 64).cast()),
			);
		}
	}
}

#[cfg_attr(target_feature = "avx512f", inline(always))]
#[cfg_attr(not(target_feature = "avx512f"), target_feature(enable = "avx512f"))]
#[cfg_attr(not(target_feature = "avx512f"), inline)]
#[cfg(any(target_feature = "avx2", test))]
unsafe fn copy_cached_block(destination: *mut u8, source: *const u8) {
	use std::arch::x86_64::_mm512_storeu_si512;
	// SAFETY: the caller admits AVX-512F and provides 256 live
	// readable/writable, disjoint bytes; these intrinsics require no alignment.
	unsafe {
		let a = _mm512_loadu_si512(source.cast());
		let b = _mm512_loadu_si512(source.add(64).cast());
		let c = _mm512_loadu_si512(source.add(128).cast());
		let d = _mm512_loadu_si512(source.add(192).cast());
		_mm512_storeu_si512(destination.cast(), a);
		_mm512_storeu_si512(destination.add(64).cast(), b);
		_mm512_storeu_si512(destination.add(128).cast(), c);
		_mm512_storeu_si512(destination.add(192).cast(), d);
	}
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn copy_streaming_avx512(
	mut destination: *mut u8,
	mut source: *const u8,
	mut length: usize,
) {
	let alignment = (64 - destination.addr() % 64) % 64;
	if alignment > 0 && alignment <= length {
		// SAFETY: the parent operation proves both ranges contain `length`
		// bytes and do not overlap; `alignment` is within that range.
		unsafe {
			super::copy_to_ptr(destination, source, alignment);
			destination = destination.add(alignment);
			source = source.add(alignment);
		}
		length -= alignment;
	}

	while length >= 256 {
		// SAFETY: the loop condition proves four 64-byte source vectors remain.
		// The destination was advanced to 64-byte alignment before streaming.
		unsafe {
			let first = _mm512_loadu_si512(source.cast());
			let second = _mm512_loadu_si512(source.add(64).cast());
			let third = _mm512_loadu_si512(source.add(128).cast());
			let fourth = _mm512_loadu_si512(source.add(192).cast());
			_mm512_stream_si512(destination.cast::<__m512i>(), first);
			_mm512_stream_si512(destination.add(64).cast::<__m512i>(), second);
			_mm512_stream_si512(destination.add(128).cast::<__m512i>(), third);
			_mm512_stream_si512(destination.add(192).cast::<__m512i>(), fourth);
			destination = destination.add(256);
			source = source.add(256);
		}
		length -= 256;
	}

	_mm_sfence();
	if length > 0 {
		// SAFETY: the parent ranges remain valid after advancing by the number of
		// bytes already copied; `length` is the exact remaining tail.
		unsafe {
			super::copy_to_ptr(destination, source, length);
		}
	}
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn copy_streaming_avx2(
	mut destination: *mut u8,
	mut source: *const u8,
	mut length: usize,
) {
	let alignment = (32 - destination.addr() % 32) % 32;
	if alignment > 0 && alignment <= length {
		// SAFETY: the parent operation proves both ranges contain `length`
		// bytes and do not overlap; `alignment` is within that range.
		unsafe {
			super::copy_to_ptr(destination, source, alignment);
			destination = destination.add(alignment);
			source = source.add(alignment);
		}
		length -= alignment;
	}

	while length >= 128 {
		// SAFETY: the loop condition proves four 32-byte source vectors remain.
		// The destination was advanced to 32-byte alignment before streaming.
		unsafe {
			let first = _mm256_loadu_si256(source.cast::<__m256i>());
			let second = _mm256_loadu_si256(source.add(32).cast::<__m256i>());
			let third = _mm256_loadu_si256(source.add(64).cast::<__m256i>());
			let fourth = _mm256_loadu_si256(source.add(96).cast::<__m256i>());
			_mm256_stream_si256(destination.cast::<__m256i>(), first);
			_mm256_stream_si256(destination.add(32).cast::<__m256i>(), second);
			_mm256_stream_si256(destination.add(64).cast::<__m256i>(), third);
			_mm256_stream_si256(destination.add(96).cast::<__m256i>(), fourth);
			destination = destination.add(128);
			source = source.add(128);
		}
		length -= 128;
	}

	// AVX2 includes the store fence instruction. It orders every non-temporal
	// store before the ordinary tail and the caller's publication.
	_mm_sfence();
	if length > 0 {
		// SAFETY: the parent ranges remain valid after advancing by the number of
		// bytes already copied; `length` is the exact remaining tail.
		unsafe {
			super::copy_to_ptr(destination, source, length);
		}
	}
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn equal_avx2(
	mut left: *const u8,
	mut right: *const u8,
	mut length: usize,
) -> bool {
	while length >= 32 {
		// SAFETY: the loop condition proves one complete vector remains in both
		// initialized source ranges. Unaligned vector loads are deliberate.
		let matches = unsafe {
			let left_vector = _mm256_loadu_si256(left.cast::<__m256i>());
			let right_vector = _mm256_loadu_si256(right.cast::<__m256i>());
			_mm256_movemask_epi8(_mm256_cmpeq_epi8(left_vector, right_vector))
		};
		if matches != -1 {
			return false;
		}
		// SAFETY: the loop condition proves the advanced pointers remain at or
		// before the end of their allocations.
		unsafe {
			left = left.add(32);
			right = right.add(32);
		}
		length -= 32;
	}

	// SAFETY: both pointers address `length` initialized remaining bytes.
	unsafe { std::slice::from_raw_parts(left, length) == std::slice::from_raw_parts(right, length) }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn equal_avx512(
	mut left: *const u8,
	mut right: *const u8,
	mut length: usize,
) -> bool {
	while length >= 64 {
		// SAFETY: the loop condition proves one complete vector remains in both
		// initialized source ranges. Unaligned vector loads are deliberate.
		let matches = unsafe {
			let left_vector = _mm512_loadu_si512(left.cast());
			let right_vector = _mm512_loadu_si512(right.cast());
			_mm512_cmpeq_epi8_mask(left_vector, right_vector)
		};
		if matches != u64::MAX {
			return false;
		}
		// SAFETY: the loop condition proves the advanced pointers remain at or
		// before the end of their allocations.
		unsafe {
			left = left.add(64);
			right = right.add(64);
		}
		length -= 64;
	}

	// SAFETY: both pointers address `length` initialized remaining bytes.
	unsafe { std::slice::from_raw_parts(left, length) == std::slice::from_raw_parts(right, length) }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn equal_constant_time_avx2(
	mut left: *const u8,
	mut right: *const u8,
	mut length: usize,
) -> bool {
	let mut vector_difference = _mm256_setzero_si256();
	while length >= 32 {
		// SAFETY: the loop condition proves one complete vector remains in both
		// initialized source ranges. Unaligned vector loads are deliberate.
		unsafe {
			let left_vector = _mm256_loadu_si256(left.cast::<__m256i>());
			let right_vector = _mm256_loadu_si256(right.cast::<__m256i>());
			vector_difference = _mm256_or_si256(
				vector_difference,
				_mm256_xor_si256(left_vector, right_vector),
			);
			left = left.add(32);
			right = right.add(32);
		}
		length -= 32;
	}

	let vector_equal = _mm256_cmpeq_epi8(vector_difference, _mm256_setzero_si256());
	let mut difference = u8::from(_mm256_movemask_epi8(vector_equal) != -1);
	for index in 0..length {
		// SAFETY: `index` is within the remaining initialized tail of both ranges.
		unsafe {
			difference |= *left.add(index) ^ *right.add(index);
		}
	}
	std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
	difference == 0
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn equal_constant_time_avx512(
	mut left: *const u8,
	mut right: *const u8,
	mut length: usize,
) -> bool {
	let mut vector_difference = _mm512_setzero_si512();
	while length >= 64 {
		// SAFETY: the loop condition proves one complete vector remains in both
		// initialized source ranges. Unaligned vector loads are deliberate.
		unsafe {
			let left_vector = _mm512_loadu_si512(left.cast());
			let right_vector = _mm512_loadu_si512(right.cast());
			vector_difference = _mm512_or_si512(
				vector_difference,
				_mm512_xor_si512(left_vector, right_vector),
			);
			left = left.add(64);
			right = right.add(64);
		}
		length -= 64;
	}

	let vector_equal = _mm512_cmpeq_epi8_mask(vector_difference, _mm512_setzero_si512());
	let mut difference = u8::from(vector_equal != u64::MAX);
	for index in 0..length {
		// SAFETY: `index` is within the remaining initialized tail of both ranges.
		unsafe {
			difference |= *left.add(index) ^ *right.add(index);
		}
	}
	std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
	difference == 0
}
