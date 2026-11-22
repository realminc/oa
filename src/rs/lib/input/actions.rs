//! `InputSystem` — named key-action registry with context dispatch.
//!
//! Donor: `oa/ui/input.h` and `oa/ui/input.cpp`.
//!
//! ```ignore
//! let mut input = InputSystem::new();
//! input.register_action(KeyAction {
//!     name: "screenshot".into(),
//!     binding: KeyBinding { key: Key::F12, modifiers: INPUT_MOD_NONE },
//!     context: InputContext::Global,
//!     allow_repeat: false,
//!     callback: None,
//! });
//! // per-frame:
//! for event in events { input.dispatch(&event); }
//! ```
//!
//! Bindings can be serialized to / loaded from a versioned YAML document on
//! donor builds that link yaml-cpp. This build has no YAML backend, so the
//! persistence entry points report [`crate::ErrorKind::MissingCapability`],
//! matching the donor's documented fallback. Application callbacks are never
//! serialized.

use super::types::{
	INPUT_MOD_CTRL, INPUT_MOD_NONE, InputEvent, InputEventKind, InputModifiers, Key,
};
use crate::{Error, Result};

/// Focus domain gating which actions a key event may reach.
///
/// `Global` actions are always active; other contexts only fire while the
/// matching focus is set through [`InputSystem::set_context`].
///
/// Donor: `oa::InputContext`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputContext {
	/// Always active.
	#[default]
	Global,
	/// Active when the node canvas has focus.
	NodeCanvas,
	/// Active when a text field has focus.
	TextInput,
	/// Active when the timeline has focus.
	Timeline,
}

/// One key-plus-modifiers trigger.
///
/// Donor: `oa::KeyBinding`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeyBinding {
	pub key: Key,
	pub modifiers: InputModifiers,
}

impl KeyBinding {
	/// Return whether `event` is a key-down matching key and modifiers exactly.
	#[must_use]
	pub fn matches(&self, event: &InputEvent) -> bool {
		event.kind == InputEventKind::KeyDown
			&& event.key == self.key
			&& event.modifiers == self.modifiers
	}
}

/// One named action: trigger, focus context, repeat policy, and callback.
///
/// Donor: `oa::KeyAction`.
pub struct KeyAction {
	pub name: String,
	pub binding: KeyBinding,
	pub context: InputContext,
	pub allow_repeat: bool,
	pub callback: Option<Box<dyn FnMut()>>,
}

impl std::fmt::Debug for KeyAction {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("KeyAction")
			.field("name", &self.name)
			.field("binding", &self.binding)
			.field("context", &self.context)
			.field("allow_repeat", &self.allow_repeat)
			.field("callback", &self.callback.is_some())
			.finish()
	}
}

/// Named key-action registry with context dispatch.
///
/// Donor: `oa::InputSystem`.
#[derive(Default)]
pub struct InputSystem {
	actions: Vec<KeyAction>,
	context: InputContext,
}

impl InputSystem {
	pub fn new() -> Self {
		Self::default()
	}

	/// Register `action`, replacing any existing action with the same name.
	pub fn register_action(&mut self, action: KeyAction) {
		if let Some(existing) = self
			.actions
			.iter_mut()
			.find(|existing| existing.name == action.name)
		{
			*existing = action;
			return;
		}
		self.actions.push(action);
	}

	/// Remove the action named `name`, if present.
	pub fn unregister_action(&mut self, name: &str) {
		if let Some(index) = self.actions.iter().position(|action| action.name == name) {
			self.actions.swap_remove(index);
		}
	}

	/// Attach or replace the callback of the action named `name`.
	pub fn set_callback(&mut self, name: &str, callback: impl FnMut() + 'static) {
		if let Some(action) = self.actions.iter_mut().find(|action| action.name == name) {
			action.callback = Some(Box::new(callback));
		}
	}

	/// Replace the trigger of the action named `name`.
	pub fn rebind(&mut self, name: &str, binding: KeyBinding) {
		if let Some(action) = self.actions.iter_mut().find(|action| action.name == name) {
			action.binding = binding;
		}
	}

	/// Set the active focus context.
	pub fn set_context(&mut self, context: InputContext) {
		self.context = context;
	}

	/// Return the active focus context.
	#[must_use]
	pub const fn context(&self) -> InputContext {
		self.context
	}

	/// Return the number of registered actions.
	#[must_use]
	pub fn action_count(&self) -> usize {
		self.actions.len()
	}

	/// Dispatch `event` to the first matching action in context.
	///
	/// Returns `true` when an action's binding matched — including the case
	/// where a repeated key event is consumed without firing because
	/// `allow_repeat` is false.
	pub fn dispatch(&mut self, event: &InputEvent) -> bool {
		if event.kind != InputEventKind::KeyDown {
			return false;
		}
		for action in &mut self.actions {
			if action.context != InputContext::Global && action.context != self.context {
				continue;
			}
			if action.binding.matches(event) {
				if event.key_repeat && !action.allow_repeat {
					return true;
				}
				if let Some(callback) = &mut action.callback {
					callback();
				}
				return true;
			}
		}
		false
	}

	/// Register the reusable Realm tool bindings.
	///
	/// Installs named callback-free actions; the application attaches each
	/// behavior through [`Self::set_callback`]:
	///
	/// | action          | binding  | context    |
	/// |-----------------|----------|------------|
	/// | `screenshot`    | F12      | Global     |
	/// | `record`        | Ctrl+R   | Global     |
	/// | `toggle_camera` | Space    | Global     |
	/// | `undo`          | Ctrl+Z   | NodeCanvas |
	/// | `redo`          | Ctrl+Y   | NodeCanvas |
	/// | `save`          | Ctrl+S   | Global     |
	/// | `fit_view`      | F        | NodeCanvas |
	pub fn register_defaults(&mut self) {
		self.register_action(KeyAction {
			name: "screenshot".into(),
			binding: KeyBinding {
				key: Key::F12,
				modifiers: INPUT_MOD_NONE,
			},
			context: InputContext::Global,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "record".into(),
			binding: KeyBinding {
				key: Key::R,
				modifiers: INPUT_MOD_CTRL,
			},
			context: InputContext::Global,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "toggle_camera".into(),
			binding: KeyBinding {
				key: Key::Space,
				modifiers: INPUT_MOD_NONE,
			},
			context: InputContext::Global,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "undo".into(),
			binding: KeyBinding {
				key: Key::Z,
				modifiers: INPUT_MOD_CTRL,
			},
			context: InputContext::NodeCanvas,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "redo".into(),
			binding: KeyBinding {
				key: Key::Y,
				modifiers: INPUT_MOD_CTRL,
			},
			context: InputContext::NodeCanvas,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "save".into(),
			binding: KeyBinding {
				key: Key::S,
				modifiers: INPUT_MOD_CTRL,
			},
			context: InputContext::Global,
			allow_repeat: false,
			callback: None,
		});
		self.register_action(KeyAction {
			name: "fit_view".into(),
			binding: KeyBinding {
				key: Key::F,
				modifiers: INPUT_MOD_NONE,
			},
			context: InputContext::NodeCanvas,
			allow_repeat: false,
			callback: None,
		});
	}

	/// Load bindings from a versioned YAML document.
	///
	/// Loading is transactional: a syntax or schema error leaves registered
	/// actions untouched. Persisted rebinds preserve existing callbacks; new
	/// actions arrive callback-free.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::InvalidArgument`] for an empty path, and
	/// [`ErrorKind::MissingCapability`] while no YAML backend is linked — the
	/// donor's documented behavior without yaml-cpp.
	///
	/// [`ErrorKind::InvalidArgument`]: crate::ErrorKind::InvalidArgument
	/// [`ErrorKind::MissingCapability`]: crate::ErrorKind::MissingCapability
	pub fn load_bindings_yaml(&mut self, path: &str) -> Result<()> {
		if path.is_empty() {
			return Err(Error::invalid_argument(
				"InputSystem::load_bindings_yaml requires a path",
			));
		}
		Err(Error::missing_capability(
			"InputSystem YAML persistence requires a YAML backend",
		))
	}

	/// Save bindings to a versioned YAML document.
	///
	/// # Errors
	///
	/// Returns [`ErrorKind::InvalidArgument`] for an empty path, and
	/// [`ErrorKind::MissingCapability`] while no YAML backend is linked.
	///
	/// [`ErrorKind::InvalidArgument`]: crate::ErrorKind::InvalidArgument
	/// [`ErrorKind::MissingCapability`]: crate::ErrorKind::MissingCapability
	pub fn save_bindings_yaml(&self, path: &str) -> Result<()> {
		if path.is_empty() {
			return Err(Error::invalid_argument(
				"InputSystem::save_bindings_yaml requires a path",
			));
		}
		Err(Error::missing_capability(
			"InputSystem YAML persistence requires a YAML backend",
		))
	}
}
