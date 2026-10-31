use crate::{Error, Result};

/// Field representation at the input/output boundary of a generic cyclic NTT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NttRepresentation {
	/// Ordinary canonical residues.
	#[default]
	Standard,
	/// Residues multiplied by R = 2^64 modulo the field prime.
	Montgomery,
}

/// Checked public prime-field and exact-order root metadata for a cyclic NTT.
///
/// Coefficients use two little-endian U32 limbs, including small fields.
/// Prime/root checks use host integer metadata; tensor execution stays on GPU.
#[derive(Clone, Debug)]
pub struct NttConfig {
	pub(super) modulus: u64,
	pub(super) root: u64,
	pub(super) length: usize,
	pub(super) representation: NttRepresentation,
}

impl NttConfig {
	/// Check an odd prime below 2^62 and a root of exact order `length`.
	///
	/// # Errors
	/// Rejects non-primes, roots of smaller order and lengths outside 2^2..=2^24.
	pub fn new(modulus: u64, root: u64, length: usize) -> Result<Self> {
		if !(3..(1_u64 << 62)).contains(&modulus)
			|| !is_prime(modulus)
			|| !(4..=(1_usize << 24)).contains(&length)
			|| !length.is_power_of_two()
			|| root == 0
			|| root >= modulus
			|| !(modulus - 1).is_multiple_of(length as u64)
			|| power_mod(root, length as u64, modulus) != 1
			|| power_mod(root, (length / 2) as u64, modulus) == 1
		{
			return Err(Error::invalid_argument(
				"NTT requires a prime below 2^62, power-of-two length 4..=2^24 and a root of exact transform order",
			));
		}
		Ok(Self {
			modulus,
			root,
			length,
			representation: NttRepresentation::Standard,
		})
	}

	/// Select the representation preserved at both transform boundaries.
	pub fn with_representation(mut self, representation: NttRepresentation) -> Self {
		self.representation = representation;
		self
	}

	/// Public field prime.
	pub const fn modulus(&self) -> u64 {
		self.modulus
	}
	/// Transform length.
	pub const fn length(&self) -> usize {
		self.length
	}
	/// Root of exact transform order.
	pub const fn root(&self) -> u64 {
		self.root
	}
	/// Input/output field representation.
	pub const fn representation(&self) -> NttRepresentation {
		self.representation
	}
}

pub(super) fn power_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
	let mut result = 1;
	while exponent != 0 {
		if exponent & 1 != 0 {
			result = ((u128::from(result) * u128::from(base)) % u128::from(modulus)) as u64;
		}
		base = ((u128::from(base) * u128::from(base)) % u128::from(modulus)) as u64;
		exponent >>= 1;
	}
	result
}

fn is_prime(value: u64) -> bool {
	for prime in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
		if value.is_multiple_of(prime) {
			return value == prime;
		}
	}
	let shift = (value - 1).trailing_zeros();
	let odd = (value - 1) >> shift;
	// First twelve prime bases: deterministic throughout the admitted range.
	// Sorenson/Webster, Strong Pseudoprimes to Twelve Prime Bases (2015).
	for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
		let base = base % value;
		if base == 0 {
			continue;
		}
		let mut residue = power_mod(base, odd, value);
		if residue == 1 || residue == value - 1 {
			continue;
		}
		let mut witness = true;
		for _ in 1..shift {
			residue = ((u128::from(residue) * u128::from(residue)) % u128::from(value)) as u64;
			if residue == value - 1 {
				witness = false;
				break;
			}
		}
		if witness {
			return false;
		}
	}
	true
}
