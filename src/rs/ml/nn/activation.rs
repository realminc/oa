//! Parameterless activation modules generated from the donor NN family.

use crate::{Matrix, Result};

use super::super::{Module, ModuleRegistry, matrix};

macro_rules! activation_module {
	($(#[$meta:meta])* $name:ident, $operation:path) => {
		$(#[$meta])*
		pub struct $name {
			registry: ModuleRegistry,
		}

		impl $name {
			/// Construct the parameterless activation module.
			pub const fn new() -> Self {
				Self {
					registry: ModuleRegistry::new(),
				}
			}

			/// Apply the activation without submitting or waiting.
			///
			/// # Errors
			///
			/// Returns an error from the underlying Matrix operation.
			pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
				$operation(input)
			}
		}

		impl Default for $name {
			fn default() -> Self {
				Self::new()
			}
		}

		impl Module for $name {
			fn forward(&self, input: &Matrix) -> Result<Matrix> {
				$name::forward(self, input)
			}

			fn registry(&self) -> &ModuleRegistry {
				&self.registry
			}
		}
	};
}

activation_module!(
	/// Rectified linear unit module.
	Relu,
	matrix::relu
);
activation_module!(
	/// Gaussian error linear unit module using OA's tanh approximation.
	Gelu,
	matrix::gelu
);
activation_module!(
	/// Sigmoid linear unit module.
	Silu,
	matrix::silu
);
