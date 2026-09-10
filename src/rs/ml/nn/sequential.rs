//! Ordered neural-network module composition.

use std::rc::Rc;

use crate::{Matrix, Result};

use super::super::{Module, ModuleRegistry};

/// Apply registered child modules in insertion order.
pub struct Sequential {
	registry: ModuleRegistry,
}

impl Sequential {
	/// Construct an empty sequence.
	pub const fn new() -> Self {
		Self {
			registry: ModuleRegistry::new(),
		}
	}

	/// Append one child under the deterministic `layer_N` name.
	///
	/// # Errors
	///
	/// Returns an error if the child handle is already registered.
	pub fn add(&mut self, module: Rc<dyn Module>) -> Result<&mut Self> {
		let name = format!("layer_{}", self.len());
		self.add_named(name, module)
	}

	/// Append one child under an explicit registration name.
	///
	/// # Errors
	///
	/// Returns an error for an invalid/duplicate name or child handle.
	pub fn add_named(
		&mut self,
		name: impl Into<String>,
		module: Rc<dyn Module>,
	) -> Result<&mut Self> {
		self.registry.register_module(name, module)?;
		Ok(self)
	}

	/// Return the number of registered children.
	pub fn len(&self) -> usize {
		self.registry.child_modules().len()
	}

	/// Return whether the sequence contains no children.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Apply every child in registration order.
	///
	/// # Errors
	///
	/// Returns the first child operation or validation failure.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let mut output = input.clone();
		for child in self.registry.child_modules() {
			output = child.forward(&output)?;
		}
		Ok(output)
	}
}

impl Default for Sequential {
	fn default() -> Self {
		Self::new()
	}
}

impl Module for Sequential {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Sequential::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
