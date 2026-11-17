//! Signed nanosecond-precision duration value.
//!
//! Donor: `oa::Duration` (`oacpp/source/cpp/include/oa/core/std/chrono.h`).
//! Unlike [`std::time::Duration`] this value is signed; animation curves and
//! other sparse timeline surfaces admit negative sample times.

/// Signed nanosecond-precision duration.
///
/// Stores `i64` nanoseconds. Checked constructors return `None` on overflow
/// instead of the donor's contract abort.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration {
	nanoseconds: i64,
}

/// `2^63` as `f64`; the largest magnitude a nanosecond count may reach before
/// it can no longer convert to `i64`.
const I64_RANGE_F64: f64 = 9_223_372_036_854_776e18;

impl Duration {
	/// Zero duration.
	pub const ZERO: Self = Self { nanoseconds: 0 };

	/// Construct from a signed nanosecond count.
	pub const fn from_nanoseconds(nanoseconds: i64) -> Self {
		Self { nanoseconds }
	}

	/// Construct from whole seconds, or `None` on overflow.
	pub const fn from_seconds(seconds: i64) -> Option<Self> {
		match seconds.checked_mul(1_000_000_000) {
			Some(nanoseconds) => Some(Self { nanoseconds }),
			None => None,
		}
	}

	/// Construct from whole milliseconds, or `None` on overflow.
	pub const fn from_milliseconds(milliseconds: i64) -> Option<Self> {
		match milliseconds.checked_mul(1_000_000) {
			Some(nanoseconds) => Some(Self { nanoseconds }),
			None => None,
		}
	}

	/// Construct from whole microseconds, or `None` on overflow.
	pub const fn from_microseconds(microseconds: i64) -> Option<Self> {
		match microseconds.checked_mul(1_000) {
			Some(nanoseconds) => Some(Self { nanoseconds }),
			None => None,
		}
	}

	/// Construct from fractional seconds, or `None` when the value is not
	/// finite or cannot be represented in `i64` nanoseconds.
	pub fn from_seconds_f64(seconds: f64) -> Option<Self> {
		let nanoseconds = seconds * 1.0e9;
		if !nanoseconds.is_finite() || nanoseconds <= -I64_RANGE_F64 || nanoseconds >= I64_RANGE_F64 {
			return None;
		}
		Some(Self {
			nanoseconds: nanoseconds as i64,
		})
	}

	/// Signed nanosecond count.
	pub const fn nanoseconds(self) -> i64 {
		self.nanoseconds
	}

	/// Truncating microsecond count.
	pub const fn microseconds(self) -> i64 {
		self.nanoseconds / 1_000
	}

	/// Truncating millisecond count.
	pub const fn milliseconds(self) -> i64 {
		self.nanoseconds / 1_000_000
	}

	/// Duration as fractional seconds.
	pub fn to_seconds(self) -> f64 {
		self.nanoseconds as f64 / 1.0e9
	}

	/// Duration as fractional milliseconds.
	pub fn to_milliseconds(self) -> f64 {
		self.nanoseconds as f64 / 1.0e6
	}

	/// Duration as fractional microseconds.
	pub fn to_microseconds(self) -> f64 {
		self.nanoseconds as f64 / 1.0e3
	}

	/// Checked addition, or `None` on overflow.
	pub const fn checked_add(self, other: Self) -> Option<Self> {
		match self.nanoseconds.checked_add(other.nanoseconds) {
			Some(nanoseconds) => Some(Self { nanoseconds }),
			None => None,
		}
	}

	/// Checked subtraction, or `None` on overflow.
	pub const fn checked_sub(self, other: Self) -> Option<Self> {
		match self.nanoseconds.checked_sub(other.nanoseconds) {
			Some(nanoseconds) => Some(Self { nanoseconds }),
			None => None,
		}
	}
}
