use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::scalar::{Float, approx, valid_tolerance};

macro_rules! vector_type {
    ($name:ident { $($field:ident),+ $(,)? } $(, $normalization_hint:meta)?) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name<T: Float = f32> {
            $(pub $field: T),+
        }

        impl<T: Float> $name<T> {
            pub fn is_finite(self) -> bool {
                true $(&& self.$field.is_finite())+
            }

            pub fn length_squared(self) -> T {
                T::ZERO $(+ self.$field * self.$field)+
            }

            pub fn length(self) -> T {
                let squared = self.length_squared();
                if squared.is_finite() && squared > T::ZERO {
                    return squared.sqrt();
                }
                if !self.is_finite() {
                    return squared.sqrt();
                }
                let magnitude = T::ZERO $(.max(self.$field.abs()))+;
                if magnitude == T::ZERO {
                    return T::ZERO;
                }
                let scaled_squared = T::ZERO $(+ (self.$field / magnitude) * (self.$field / magnitude))+;
                magnitude * scaled_squared.sqrt()
            }

            /// Returns a unit vector, or `None` for a zero-length or non-finite value.
            $(#[$normalization_hint])?
            pub fn try_normalized(self) -> Option<Self> {
                let squared = self.length_squared();
                if squared.is_finite() && squared > T::ZERO {
                    return Some(self / squared.sqrt());
                }
                if !self.is_finite() {
                    return None;
                }
                let magnitude = T::ZERO $(.max(self.$field.abs()))+;
                if magnitude <= T::ZERO {
                    return None;
                }
                let scaled = self / magnitude;
                let scaled_length = scaled.length();
                if !scaled_length.is_finite() || scaled_length <= T::ZERO {
                    return None;
                }
                let result = scaled / scaled_length;
                result.is_finite().then_some(result)
            }

            pub fn dot(self, other: Self) -> T {
                T::ZERO $(+ self.$field * other.$field)+
            }

            pub fn lerp(self, other: Self, amount: T) -> Self {
                self * (T::ONE - amount) + other * amount
            }

            pub fn distance_squared(self, other: Self) -> T {
                (other - self).length_squared()
            }

            pub fn distance(self, other: Self) -> T {
                (other - self).length()
            }

            pub fn abs(self) -> Self {
                Self { $($field: self.$field.abs()),+ }
            }

            pub fn component_min(self) -> T {
                let mut result = T::from_f64(f64::INFINITY);
                $(result = result.min(self.$field);)+
                result
            }

            pub fn component_max(self) -> T {
                let mut result = T::from_f64(f64::NEG_INFINITY);
                $(result = result.max(self.$field);)+
                result
            }

            pub fn component_mul(self, other: Self) -> Self {
                Self { $($field: self.$field * other.$field),+ }
            }

            /// Divides components with ordinary IEEE floating-point behavior.
            pub fn component_div(self, other: Self) -> Self {
                Self { $($field: self.$field / other.$field),+ }
            }

            pub fn min(self, other: Self) -> Self {
                Self { $($field: self.$field.min(other.$field)),+ }
            }

            pub fn max(self, other: Self) -> Self {
                Self { $($field: self.$field.max(other.$field)),+ }
            }

            pub fn clamp(self, minimum: Self, maximum: Self) -> Self {
                self.max(minimum).min(maximum)
            }

            pub fn approximately_equal(self, other: Self, absolute: T, relative: T) -> bool {
                valid_tolerance(absolute)
                    && valid_tolerance(relative)
                    $(&& approx(self.$field, other.$field, absolute, relative))+
            }
        }

        impl<T: Float> Add for $name<T> {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                Self { $($field: self.$field + other.$field),+ }
            }
        }

        impl<T: Float> Sub for $name<T> {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                Self { $($field: self.$field - other.$field),+ }
            }
        }

        impl<T: Float> Mul<T> for $name<T> {
            type Output = Self;

            fn mul(self, scalar: T) -> Self {
                Self { $($field: self.$field * scalar),+ }
            }
        }

        impl<T: Float> Div<T> for $name<T> {
            type Output = Self;

            fn div(self, scalar: T) -> Self {
                Self { $($field: self.$field / scalar),+ }
            }
        }

        impl<T: Float> Neg for $name<T> {
            type Output = Self;

            fn neg(self) -> Self {
                Self { $($field: -self.$field),+ }
            }
        }

        impl<T: Float> AddAssign for $name<T> {
            fn add_assign(&mut self, other: Self) {
                *self = *self + other;
            }
        }

        impl<T: Float> SubAssign for $name<T> {
            fn sub_assign(&mut self, other: Self) {
                *self = *self - other;
            }
        }

        impl<T: Float> MulAssign<T> for $name<T> {
            fn mul_assign(&mut self, scalar: T) {
                *self = *self * scalar;
            }
        }

        impl<T: Float> DivAssign<T> for $name<T> {
            fn div_assign(&mut self, scalar: T) {
                *self = *self / scalar;
            }
        }
    };
}

vector_type!(Vec2 { x, y });
vector_type!(Vec3 { x, y, z }, inline);
vector_type!(Vec4 { x, y, z, w }, inline);

pub type DVec2 = Vec2<f64>;
pub type DVec3 = Vec3<f64>;
pub type DVec4 = Vec4<f64>;

impl<T: Float> Vec3<T> {
	pub fn cross(self, other: Self) -> Self {
		Self {
			x: self.y * other.z - self.z * other.y,
			y: self.z * other.x - self.x * other.z,
			z: self.x * other.y - self.y * other.x,
		}
	}

	pub fn reflect(self, normal: Self) -> Self {
		self - normal * (T::TWO * normal.dot(self))
	}

	pub fn refract(self, normal: Self, eta: T) -> Self {
		let normal_dot_incident = normal.dot(self);
		let discriminant = T::ONE - eta * eta * (T::ONE - normal_dot_incident * normal_dot_incident);
		if discriminant < T::ZERO {
			return Self::default();
		}
		self * eta - normal * (eta * normal_dot_incident + discriminant.sqrt())
	}

	pub fn face_forward(self, incident: Self, reference_normal: Self) -> Self {
		if reference_normal.dot(incident) < T::ZERO {
			self
		} else {
			-self
		}
	}

	#[inline]
	pub fn try_project_onto(self, onto: Self, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !self.is_finite() {
			return None;
		}
		let denominator = onto.dot(onto);
		let squared_tolerance = tolerance * tolerance;
		if denominator.is_finite() && squared_tolerance.is_finite() && denominator > squared_tolerance {
			let factor = self.dot(onto) / denominator;
			let projection = onto * factor;
			if factor.is_finite() && projection.is_finite() {
				return Some(projection);
			}
		}
		self.project_onto_scaled(onto)
	}

	#[cold]
	#[inline(never)]
	fn project_onto_scaled(self, onto: Self) -> Option<Self> {
		let onto = onto.try_normalized()?;
		let magnitude = self.x.abs().max(self.y.abs()).max(self.z.abs());
		let projection = if magnitude > T::ZERO {
			onto * ((self / magnitude).dot(onto) * magnitude)
		} else {
			Self::default()
		};
		projection.is_finite().then_some(projection)
	}

	pub fn try_reject_from(self, from: Self, tolerance: T) -> Option<Self> {
		let result = self - self.try_project_onto(from, tolerance)?;
		result.is_finite().then_some(result)
	}

	pub fn try_angle_between(self, other: Self, tolerance: T) -> Option<T> {
		if !valid_tolerance(tolerance) {
			return None;
		}
		let a = self.try_normalized()?;
		let b = other.try_normalized()?;
		if self.length() <= tolerance || other.length() <= tolerance {
			return None;
		}
		let angle = a.dot(b).max(-T::ONE).min(T::ONE).acos();
		angle.is_finite().then_some(angle)
	}

	pub fn try_perpendicular(self, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || self.length() <= tolerance {
			return None;
		}
		let direction = self.try_normalized()?;
		let candidate =
			if direction.x.abs() <= direction.y.abs() && direction.x.abs() <= direction.z.abs() {
				Self {
					x: T::ONE,
					y: T::ZERO,
					z: T::ZERO,
				}
			} else if direction.y.abs() <= direction.z.abs() {
				Self {
					x: T::ZERO,
					y: T::ONE,
					z: T::ZERO,
				}
			} else {
				Self {
					x: T::ZERO,
					y: T::ZERO,
					z: T::ONE,
				}
			};
		direction.cross(candidate).try_normalized()
	}

	pub fn try_signed_angle_between(self, other: Self, axis: Self, tolerance: T) -> Option<T> {
		if !valid_tolerance(tolerance)
			|| self.length() <= tolerance
			|| other.length() <= tolerance
			|| axis.length() <= tolerance
		{
			return None;
		}
		let a = self.try_normalized()?;
		let b = other.try_normalized()?;
		let axis = axis.try_normalized()?;
		let sine = axis.dot(a.cross(b));
		let cosine = a.dot(b).max(-T::ONE).min(T::ONE);
		let angle = sine.atan2(cosine);
		angle.is_finite().then_some(angle)
	}
}

macro_rules! scalar_left_mul {
	($scalar:ty, $vector:ident) => {
		impl Mul<$vector<$scalar>> for $scalar {
			type Output = $vector<$scalar>;

			fn mul(self, vector: $vector<$scalar>) -> Self::Output {
				vector * self
			}
		}
	};
}

scalar_left_mul!(f32, Vec2);
scalar_left_mul!(f32, Vec3);
scalar_left_mul!(f32, Vec4);
scalar_left_mul!(f64, Vec2);
scalar_left_mul!(f64, Vec3);
scalar_left_mul!(f64, Vec4);
