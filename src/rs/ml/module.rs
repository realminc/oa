use std::{
	cell::{Cell, RefCell},
	rc::Rc,
};

use crate::{Error, Matrix, Result};

use super::Parameter;

/// Owned dotted path and stable trainable parameter handle.
pub struct NamedParameter {
	path: String,
	parameter: Parameter,
}

impl NamedParameter {
	/// Return the registration-derived dotted path.
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Return the stable parameter handle.
	pub fn parameter(&self) -> Parameter {
		self.parameter.clone()
	}
}

/// Owned dotted path and non-trainable Matrix buffer.
pub struct NamedBuffer {
	path: String,
	data: Rc<RefCell<Matrix>>,
	persistent: bool,
}

/// Owned dotted path and host scalar that participates in module persistence.
///
/// Scalar state is intended for small algorithmic counters and flags whose
/// value affects the next device operation but does not belong in a Matrix
/// buffer. It is traversed and checkpointed with the same deterministic module
/// tree as parameters and buffers.
pub struct NamedStateU32 {
	path: String,
	value: Rc<Cell<u32>>,
}

impl NamedStateU32 {
	/// Return the registration-derived dotted path.
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Return the current scalar value.
	pub fn value(&self) -> u32 {
		self.value.get()
	}

	pub(crate) fn replace_value(&self, value: u32) {
		self.value.set(value);
	}
}

impl NamedBuffer {
	/// Return the registration-derived dotted path.
	pub fn path(&self) -> &str {
		&self.path
	}

	/// Return a cheap handle to the registered value.
	pub fn data(&self) -> Matrix {
		self.data.borrow().clone()
	}

	/// Return whether persistence must include this buffer.
	pub const fn persistent(&self) -> bool {
		self.persistent
	}

	pub(crate) fn replace_data(&self, data: Matrix) -> Result<()> {
		let current = self.data.borrow();
		if current.shape() != data.shape()
			|| current.dtype() != data.dtype()
			|| !current.engine_handle().same_as(data.engine_handle())
		{
			return Err(Error::invalid_argument(format!(
				"replacement data does not match module buffer {}",
				self.path
			)));
		}
		drop(current);
		*self.data.borrow_mut() = data;
		Ok(())
	}
}

struct ParameterEntry {
	name: String,
	parameter: Parameter,
}

struct BufferEntry {
	name: String,
	data: Rc<RefCell<Matrix>>,
	persistent: bool,
}

struct StateU32Entry {
	name: String,
	value: Rc<Cell<u32>>,
}

struct ChildEntry {
	name: String,
	module: Rc<dyn Module>,
}

/// Constructor-owned structural registry for one [`Module`].
///
/// Registration requires `&mut self`, so a module's parameter/buffer/child tree
/// is fixed before the module is shared through `Rc`.
pub struct ModuleRegistry {
	parameters: Vec<ParameterEntry>,
	buffers: Vec<BufferEntry>,
	state_u32: Vec<StateU32Entry>,
	children: Vec<ChildEntry>,
	training: Cell<bool>,
}

impl ModuleRegistry {
	/// Construct an empty registry in training mode.
	pub const fn new() -> Self {
		Self {
			parameters: Vec::new(),
			buffers: Vec::new(),
			state_u32: Vec::new(),
			children: Vec::new(),
			training: Cell::new(true),
		}
	}

	/// Register one direct parameter.
	///
	/// # Errors
	///
	/// Returns an error for an invalid/duplicate local name or when the same
	/// parameter handle is already registered directly.
	pub fn register_parameter(
		&mut self,
		name: impl Into<String>,
		parameter: Parameter,
	) -> Result<()> {
		let name = self.validate_new_name(name.into())?;
		if self
			.parameters
			.iter()
			.any(|entry| entry.parameter.same_as(&parameter))
		{
			return Err(Error::invalid_argument(
				"module parameter handle is already registered",
			));
		}
		self.parameters.push(ParameterEntry { name, parameter });
		Ok(())
	}

	/// Register one direct non-trainable buffer.
	///
	/// # Errors
	///
	/// Returns an error for an invalid/duplicate local name or duplicate semantic
	/// Matrix value.
	pub fn register_buffer(
		&mut self,
		name: impl Into<String>,
		data: Matrix,
		persistent: bool,
	) -> Result<()> {
		let name = self.validate_new_name(name.into())?;
		if self
			.buffers
			.iter()
			.any(|entry| entry.data.borrow().same_value_as(&data))
		{
			return Err(Error::invalid_argument(
				"module buffer value is already registered",
			));
		}
		self.buffers.push(BufferEntry {
			name,
			data: Rc::new(RefCell::new(data)),
			persistent,
		});
		Ok(())
	}

	/// Register one direct persistent `u32` scalar state value.
	///
	/// This is for compact host metadata whose value changes the next operation,
	/// such as an EMA transition counter. Bulk numerical state remains a Matrix
	/// buffer.
	///
	/// # Errors
	///
	/// Returns an error for an invalid or duplicate local name.
	pub fn register_state_u32(
		&mut self,
		name: impl Into<String>,
		value: u32,
	) -> Result<NamedStateU32> {
		let name = self.validate_new_name(name.into())?;
		let value = Rc::new(Cell::new(value));
		self.state_u32.push(StateU32Entry {
			name: name.clone(),
			value: value.clone(),
		});
		Ok(NamedStateU32 { path: name, value })
	}

	/// Register one owned child handle.
	///
	/// The child immediately inherits this registry's train/eval mode.
	///
	/// # Errors
	///
	/// Returns an error for an invalid/duplicate local name or when the same child
	/// handle is already registered directly.
	pub fn register_module(
		&mut self,
		name: impl Into<String>,
		module: Rc<dyn Module>,
	) -> Result<()> {
		let name = self.validate_new_name(name.into())?;
		if self
			.children
			.iter()
			.any(|entry| Rc::ptr_eq(&entry.module, &module))
		{
			return Err(Error::invalid_argument(
				"module child handle is already registered",
			));
		}
		module.train(self.training.get());
		self.children.push(ChildEntry { name, module });
		Ok(())
	}

	fn validate_new_name(&self, name: String) -> Result<String> {
		let mut bytes = name.bytes();
		let first = bytes.next();
		let valid = first.is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
			&& bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
		if !valid {
			return Err(Error::invalid_argument(format!(
				"module registration name {name:?} must be a nonempty ASCII identifier without dots"
			)));
		}
		if self.parameters.iter().any(|entry| entry.name == name)
			|| self.buffers.iter().any(|entry| entry.name == name)
			|| self.state_u32.iter().any(|entry| entry.name == name)
			|| self.children.iter().any(|entry| entry.name == name)
		{
			return Err(Error::invalid_argument(format!(
				"module registration name {name:?} is already used"
			)));
		}
		Ok(name)
	}

	pub(crate) fn child_modules(&self) -> impl ExactSizeIterator<Item = &Rc<dyn Module>> {
		self.children.iter().map(|entry| &entry.module)
	}

	pub(crate) fn buffer_handle(&self, name: &str) -> Option<NamedBuffer> {
		self.buffers
			.iter()
			.find(|entry| entry.name == name)
			.map(|entry| NamedBuffer {
				path: entry.name.clone(),
				data: entry.data.clone(),
				persistent: entry.persistent,
			})
	}
}

impl Default for ModuleRegistry {
	fn default() -> Self {
		Self::new()
	}
}

/// Stateful neural-network component with one registered ownership tree.
pub trait Module {
	/// Evaluate one Matrix input without submitting or waiting.
	fn forward(&self, input: &Matrix) -> Result<Matrix>;

	/// Return this module's structural registry.
	fn registry(&self) -> &ModuleRegistry;

	/// Return direct parameters only.
	fn parameters(&self) -> Vec<Parameter> {
		self.registry()
			.parameters
			.iter()
			.map(|entry| entry.parameter.clone())
			.collect()
	}

	/// Return direct parameters with local registration names.
	fn named_parameters(&self) -> Vec<NamedParameter> {
		self.registry()
			.parameters
			.iter()
			.map(|entry| NamedParameter {
				path: entry.name.clone(),
				parameter: entry.parameter.clone(),
			})
			.collect()
	}

	/// Return direct non-trainable buffers only.
	fn buffers(&self) -> Vec<Matrix> {
		self.registry()
			.buffers
			.iter()
			.map(|entry| entry.data.borrow().clone())
			.collect()
	}

	/// Return direct buffers with local registration names and persistence flags.
	fn named_buffers(&self) -> Vec<NamedBuffer> {
		self.registry()
			.buffers
			.iter()
			.map(|entry| NamedBuffer {
				path: entry.name.clone(),
				data: entry.data.clone(),
				persistent: entry.persistent,
			})
			.collect()
	}

	/// Return every parameter exactly once in deterministic depth-first order.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` if independently constructed registries expose
	/// one shared parameter through multiple tree paths.
	fn all_parameters(&self) -> Result<Vec<Parameter>> {
		Ok(collect_named_parameters(self.registry())?
			.into_iter()
			.map(|entry| entry.parameter)
			.collect())
	}

	/// Return every parameter with its registration-derived dotted path.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` for duplicate parameter identity.
	fn all_named_parameters(&self) -> Result<Vec<NamedParameter>> {
		collect_named_parameters(self.registry())
	}

	/// Return all direct and recursive buffers in deterministic depth-first order.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` for duplicate child or buffer identity.
	fn all_named_buffers(&self) -> Result<Vec<NamedBuffer>> {
		let mut output = Vec::new();
		let mut modules = Vec::new();
		collect_named_buffers(self.registry(), "", &mut modules, &mut output)?;
		Ok(output)
	}

	/// Return all direct and recursive persistent `u32` scalar state in
	/// deterministic depth-first order.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` for duplicate child ownership.
	fn all_named_state_u32(&self) -> Result<Vec<NamedStateU32>> {
		let mut output = Vec::new();
		let mut modules = Vec::new();
		collect_named_state_u32(self.registry(), "", &mut modules, &mut output)?;
		Ok(output)
	}

	/// Return the total number of trainable scalar values.
	///
	/// # Errors
	///
	/// Returns an error for duplicate parameter identity or count overflow.
	fn num_parameters(&self) -> Result<usize> {
		self.all_parameters()?
			.into_iter()
			.try_fold(0_usize, |total, parameter| {
				total
					.checked_add(parameter.data().num_elements())
					.ok_or_else(|| {
						Error::resource_exhausted("module parameter count overflows usize")
					})
			})
	}

	/// Set train/eval mode recursively.
	fn train(&self, training: bool) {
		self.registry().training.set(training);
		for child in &self.registry().children {
			child.module.train(training);
		}
	}

	/// Enter evaluation mode recursively.
	fn eval(&self) {
		self.train(false);
	}

	/// Return this module's current train/eval mode.
	fn is_training(&self) -> bool {
		self.registry().training.get()
	}

	/// Enter evaluation mode until the returned guard is dropped.
	fn scoped_eval(&self) -> ScopedEval<'_> {
		ScopedEval::new(self.registry())
	}
}

/// Bounded evaluation mode that restores the complete module tree on drop.
#[must_use]
pub struct ScopedEval<'a> {
	registry: &'a ModuleRegistry,
	was_training: bool,
}

impl<'a> ScopedEval<'a> {
	fn new(registry: &'a ModuleRegistry) -> Self {
		let was_training = registry.training.get();
		set_training(registry, false);
		Self {
			registry,
			was_training,
		}
	}
}

impl Drop for ScopedEval<'_> {
	fn drop(&mut self) {
		set_training(self.registry, self.was_training);
	}
}

fn collect_named_parameters(registry: &ModuleRegistry) -> Result<Vec<NamedParameter>> {
	let mut output = Vec::new();
	let mut modules = Vec::new();
	collect_parameters(registry, "", &mut modules, &mut output)?;
	Ok(output)
}

fn collect_parameters(
	registry: &ModuleRegistry,
	prefix: &str,
	modules: &mut Vec<*const ModuleRegistry>,
	output: &mut Vec<NamedParameter>,
) -> Result<()> {
	register_tree_node(registry, prefix, modules)?;
	for entry in &registry.parameters {
		if output
			.iter()
			.any(|existing| existing.parameter.same_as(&entry.parameter))
		{
			return Err(Error::failed_precondition(format!(
				"parameter registered through multiple module paths, including {}",
				join_path(prefix, &entry.name)
			)));
		}
		output.push(NamedParameter {
			path: join_path(prefix, &entry.name),
			parameter: entry.parameter.clone(),
		});
	}
	for child in &registry.children {
		collect_parameters(
			child.module.registry(),
			&join_path(prefix, &child.name),
			modules,
			output,
		)?;
	}
	Ok(())
}

fn collect_named_buffers(
	registry: &ModuleRegistry,
	prefix: &str,
	modules: &mut Vec<*const ModuleRegistry>,
	output: &mut Vec<NamedBuffer>,
) -> Result<()> {
	register_tree_node(registry, prefix, modules)?;
	for entry in &registry.buffers {
		if output
			.iter()
			.any(|existing| existing.data.borrow().same_value_as(&entry.data.borrow()))
		{
			return Err(Error::failed_precondition(format!(
				"buffer registered through multiple module paths, including {}",
				join_path(prefix, &entry.name)
			)));
		}
		output.push(NamedBuffer {
			path: join_path(prefix, &entry.name),
			data: entry.data.clone(),
			persistent: entry.persistent,
		});
	}
	for child in &registry.children {
		collect_named_buffers(
			child.module.registry(),
			&join_path(prefix, &child.name),
			modules,
			output,
		)?;
	}
	Ok(())
}

fn collect_named_state_u32(
	registry: &ModuleRegistry,
	prefix: &str,
	modules: &mut Vec<*const ModuleRegistry>,
	output: &mut Vec<NamedStateU32>,
) -> Result<()> {
	register_tree_node(registry, prefix, modules)?;
	for entry in &registry.state_u32 {
		output.push(NamedStateU32 {
			path: join_path(prefix, &entry.name),
			value: entry.value.clone(),
		});
	}
	for child in &registry.children {
		collect_named_state_u32(
			child.module.registry(),
			&join_path(prefix, &child.name),
			modules,
			output,
		)?;
	}
	Ok(())
}

fn register_tree_node(
	registry: &ModuleRegistry,
	path: &str,
	modules: &mut Vec<*const ModuleRegistry>,
) -> Result<()> {
	let identity = std::ptr::from_ref(registry);
	if modules.contains(&identity) {
		let path = if path.is_empty() { "<root>" } else { path };
		return Err(Error::failed_precondition(format!(
			"module registered through multiple tree paths, including {path}"
		)));
	}
	modules.push(identity);
	Ok(())
}

fn set_training(registry: &ModuleRegistry, training: bool) {
	registry.training.set(training);
	for child in &registry.children {
		set_training(child.module.registry(), training);
	}
}

fn join_path(prefix: &str, name: &str) -> String {
	if prefix.is_empty() {
		name.to_owned()
	} else {
		format!("{prefix}.{name}")
	}
}
