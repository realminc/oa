//! Fault on any access across the exact source or destination allocation edge.
#![cfg(all(target_os = "linux", not(miri)))]

use oa::core::memory;

struct Guarded {
	base: *mut libc::c_void,
	page: usize,
}

impl Guarded {
	fn new() -> Self {
		// SAFETY: sysconf has no pointer arguments. Anonymous mmap creates an
		// independently owned mapping; failure is checked before pointer use.
		let (page, base) = unsafe {
			let page = usize::try_from(libc::sysconf(libc::_SC_PAGESIZE)).unwrap();
			assert!(page >= 4096);
			let length = page.checked_mul(4).unwrap();
			let base = libc::mmap(
				std::ptr::null_mut(),
				length,
				libc::PROT_NONE,
				libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
				-1,
				0,
			);
			assert_ne!(base, libc::MAP_FAILED);
			(page, base)
		};
		let owner = Self { base, page };
		// SAFETY: the middle two pages belong to this live mapping and are page
		// aligned. Both outer pages remain inaccessible. Initialize raw storage
		// before exposing a Rust byte slice.
		unsafe {
			let data = base.cast::<u8>().add(page);
			assert_eq!(
				libc::mprotect(data.cast(), 2 * page, libc::PROT_READ | libc::PROT_WRITE),
				0
			);
			std::ptr::write_bytes(data, 0, 2 * page);
		}
		owner
	}

	fn bytes(&mut self) -> &mut [u8] {
		// SAFETY: the initialized writable middle pages remain mapped for this
		// exclusive borrow of their unique owner.
		unsafe {
			std::slice::from_raw_parts_mut(self.base.cast::<u8>().add(self.page), 2 * self.page)
		}
	}
}

impl Drop for Guarded {
	fn drop(&mut self) {
		// SAFETY: this owner releases exactly the mapping it created; borrows of
		// its bytes have ended before destruction.
		unsafe { assert_eq!(libc::munmap(self.base, 4 * self.page), 0) };
	}
}

#[test]
fn copies_do_not_touch_guard_pages_or_neighboring_bytes() {
	let mut source = Guarded::new();
	let mut destination = Guarded::new();
	for (index, byte) in source.bytes().iter_mut().enumerate() {
		*byte = index.wrapping_mul(131).wrapping_add(17) as u8;
	}
	for length in (0..=1088).chain([2047, 2048, 2049, 4095, 4096, 4097, 8192]) {
		for source_at_end in [false, true] {
			for destination_at_end in [false, true] {
				let source = source.bytes();
				let destination = destination.bytes();
				let src_start = if source_at_end {
					source.len() - length
				} else {
					0
				};
				let dst_start = if destination_at_end {
					destination.len() - length
				} else {
					0
				};
				let src = &source[src_start..src_start + length];
				for streaming in [false, true] {
					destination.fill(0xa5);
					let dst = &mut destination[dst_start..dst_start + length];
					if streaming {
						memory::copy_streaming(dst, src).unwrap();
					} else {
						memory::copy(dst, src).unwrap();
					}
					assert_eq!(dst, src, "length={length}, streaming={streaming}");
					assert!(destination[..dst_start].iter().all(|&byte| byte == 0xa5));
					assert!(
						destination[dst_start + length..]
							.iter()
							.all(|&byte| byte == 0xa5)
					);
				}
			}
		}
	}
}
