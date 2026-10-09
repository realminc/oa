//! Backend-neutral RGBA color value.
//!
//! Donor: `oacpp/source/cpp/include/oa/core/color.h` and `fnColor.h`.
//! `Color` carries four floating-point components but no encoded transfer
//! function or gamut. Domain values such as video colorimetry and HDR emission
//! retain their own typed metadata.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Four-component RGBA color used by UI, plotting, overlays, and ordinary
/// normalized render colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
	pub r: f32,
	pub g: f32,
	pub b: f32,
	pub a: f32,
}

impl Color {
	pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
		Self { r, g, b, a }
	}

	pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
		Self::new(r, g, b, 1.0)
	}

	/// Pack to `0xRRGGBBAA`, clamping each component and rounding to nearest.
	pub fn to_u32(self) -> u32 {
		let to_u8 = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u32;
		(to_u8(self.r) << 24) | (to_u8(self.g) << 16) | (to_u8(self.b) << 8) | to_u8(self.a)
	}

	/// Unpack one `0xRRGGBBAA` word.
	pub const fn from_u32(rgba: u32) -> Self {
		Self::new(
			((rgba >> 24) & 0xff) as f32 / 255.0,
			((rgba >> 16) & 0xff) as f32 / 255.0,
			((rgba >> 8) & 0xff) as f32 / 255.0,
			(rgba & 0xff) as f32 / 255.0,
		)
	}

	pub const fn with_alpha(self, alpha: f32) -> Self {
		Self::new(self.r, self.g, self.b, alpha)
	}

	pub const fn lerp(self, other: Self, fraction: f32) -> Self {
		Self::new(
			self.r + (other.r - self.r) * fraction,
			self.g + (other.g - self.g) * fraction,
			self.b + (other.b - self.b) * fraction,
			self.a + (other.a - self.a) * fraction,
		)
	}

	pub fn clamp(self) -> Self {
		Self::new(
			self.r.clamp(0.0, 1.0),
			self.g.clamp(0.0, 1.0),
			self.b.clamp(0.0, 1.0),
			self.a.clamp(0.0, 1.0),
		)
	}

	pub const fn to_array(self) -> [f32; 4] {
		[self.r, self.g, self.b, self.a]
	}

	pub const fn accent() -> Self {
		Self::new(0.388, 0.400, 0.945, 1.0)
	}
	pub const fn accent_hover() -> Self {
		Self::new(0.506, 0.549, 0.973, 1.0)
	}
	pub const fn success() -> Self {
		Self::new(0.188, 0.820, 0.345, 1.0)
	}
	pub const fn warning() -> Self {
		Self::new(0.961, 0.620, 0.043, 1.0)
	}
	pub const fn error() -> Self {
		Self::new(1.000, 0.271, 0.227, 1.0)
	}
	pub const fn orange() -> Self {
		Self::new(1.000, 0.420, 0.208, 1.0)
	}
	pub const fn purple() -> Self {
		Self::new(0.659, 0.333, 0.969, 1.0)
	}
	pub const fn cyan() -> Self {
		Self::new(0.133, 0.827, 0.933, 1.0)
	}
	pub const fn pink() -> Self {
		Self::new(0.925, 0.282, 0.600, 1.0)
	}
	pub const fn yellow() -> Self {
		Self::warning()
	}
	pub const fn text_primary() -> Self {
		Self::new(0.961, 0.961, 0.961, 1.0)
	}
}

impl Default for Color {
	fn default() -> Self {
		Self::new(0.0, 0.0, 0.0, 1.0)
	}
}

macro_rules! component_binary {
	($trait:ident, $method:ident, $op:tt) => {
		impl $trait for Color {
			type Output = Self;
			fn $method(self, rhs: Self) -> Self { Self::new(self.r $op rhs.r, self.g $op rhs.g, self.b $op rhs.b, self.a $op rhs.a) }
		}
	};
}
component_binary!(Add, add, +);
component_binary!(Sub, sub, -);
component_binary!(Mul, mul, *);

impl Mul<f32> for Color {
	type Output = Self;
	fn mul(self, rhs: f32) -> Self {
		Self::new(self.r * rhs, self.g * rhs, self.b * rhs, self.a * rhs)
	}
}
impl Mul<Color> for f32 {
	type Output = Color;
	fn mul(self, rhs: Color) -> Color {
		rhs * self
	}
}
impl Div<f32> for Color {
	type Output = Self;
	fn div(self, rhs: f32) -> Self {
		Self::new(self.r / rhs, self.g / rhs, self.b / rhs, self.a / rhs)
	}
}

macro_rules! assign_op {
	($trait:ident, $method:ident, $op:tt, $rhs:ty) => {
		impl $trait<$rhs> for Color {
			fn $method(&mut self, rhs: $rhs) { *self = *self $op rhs; }
		}
	};
}
assign_op!(AddAssign, add_assign, +, Color);
assign_op!(SubAssign, sub_assign, -, Color);
assign_op!(MulAssign, mul_assign, *, Color);
assign_op!(MulAssign, mul_assign, *, f32);
assign_op!(DivAssign, div_assign, /, f32);
