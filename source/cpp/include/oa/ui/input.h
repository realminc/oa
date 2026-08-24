// Ui — InputSystem: named key-action registry with context dispatch.
//
// usage:
//   InputSystem input;
//   input.registerAction({
//       .name    = "screenshot",
//       .binding = {.key = UiKey::F12, .modifiers = UiModifierNone},
//       .context = InputContext::Global,
//   });
//   // per-frame:
//   for (auto& e : events) input.dispatch(e);
//
// Bindings can be serialized to / loaded from a versioned YAML document.
// Application callbacks are never serialized. Loading updates an existing
// action without replacing its callback and admits new callback-free actions
// for applications to attach afterward.

#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/ui/event.h>


// ─── InputContext ───────────────────────────────────────────────────────────

namespace oa {

enum class InputContext : oa::U8 {
	Global     = 0,  // always active
	NodeCanvas = 1,  // active when canvas has focus
	TextInput  = 2,  // active when a text field has focus
	Timeline   = 3,
};


// ─── KeyBinding ─────────────────────────────────────────────────────────────

struct KeyBinding {
	UiKey key       = UiKey::Unknown;
	oa::U32  modifiers = UiModifierNone;

	[[nodiscard]] bool matches(const UiEvent& inEvent) const noexcept {
		return inEvent.type == UiEventType::KeyDown
		    and inEvent.key == key
		    and inEvent.modifiers == modifiers;
	}
};


// ─── KeyAction ──────────────────────────────────────────────────────────────

struct KeyAction {
	oa::String       name;
	KeyBinding   binding;
	InputContext context  = InputContext::Global;
	bool           allowRepeat = false;
	oa::Fn<void()> callback;
};


// ─── InputSystem ────────────────────────────────────────────────────────────

class InputSystem {
public:
	InputSystem() = default;
	InputSystem(const InputSystem&)            = delete;
	InputSystem& operator=(const InputSystem&) = delete;
	InputSystem(InputSystem&&) noexcept        = default;
	InputSystem& operator=(InputSystem&&) noexcept = default;

	void registerAction(KeyAction inAction);
	void unregisterAction(oa::StringView inName);
	void setCallback(oa::StringView inName, oa::Fn<void()> inCallback);
	void rebind(oa::StringView inName, KeyBinding inBinding);

	void setContext(InputContext inCtx) noexcept { context_ = inCtx; }
	[[nodiscard]] InputContext context() const noexcept { return context_; }

	// Returns true if inEvent was consumed by a matching action.
	[[nodiscard]] bool dispatch(const UiEvent& inEvent);

	// register the reusable Realm tool bindings. This installs named actions
	// with no callbacks; the application owns every operation they trigger:
	//   F12       — screenshot
	//   ctrl+R    — start/stop recording
	//   Space     — toggle camera
	//   ctrl+Z    — undo (node graph)
	//   ctrl+Y    — redo
	//   ctrl+S    — save
	//   F         — fit-to-view (node canvas)
	void registerDefaults();

	// YAML schema: {version: 1, bindings: [{action, key, modifiers,
	// context, allow_repeat}]}. Loading is transactional for syntax and schema
	// errors. Without yaml-cpp these calls report Unimplemented, never success.
	[[nodiscard]] oa::Status loadBindingsYaml(oa::StringView inPath);
	[[nodiscard]] oa::Status saveBindingsYaml(oa::StringView inPath) const;

private:
	oa::Vec<KeyAction> actions_;
	InputContext     context_ = InputContext::Global;
};

}  // namespace oa
