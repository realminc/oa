// OaUi — OaInputSystem: named key-action registry with context dispatch.
//
// Usage:
//   OaInputSystem input;
//   input.RegisterAction({
//       .Name    = "screenshot",
//       .Binding = {.Key = OuiKey::F12, .Modifiers = OUI_MOD_NONE},
//       .Context = OaInputContext::Global,
//       .Callback = [&] { OaScreenshot("", rgba_data, w, h); },
//   });
//   // per-frame:
//   for (auto& e : events) input.Dispatch(e);
//
// Bindings can be serialized to / loaded from YAML (same prefs file as
// OaEditorPrefs in the editor repo).

#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Ui/Event.h>


// ─── OaInputContext ───────────────────────────────────────────────────────────

enum class OaInputContext : OaU8 {
	Global     = 0,  // always active
	NodeCanvas = 1,  // active when canvas has focus
	TextInput  = 2,  // active when a text field has focus
	Timeline   = 3,
};


// ─── OaKeyBinding ─────────────────────────────────────────────────────────────

struct OaKeyBinding {
	OuiKey Key       = OuiKey::Unknown;
	OaU32  Modifiers = OUI_MOD_NONE;

	[[nodiscard]] bool Matches(const OaUiEvent& InEvent) const noexcept {
		return InEvent.Type == OuiEventType::KeyDown
		    and InEvent.Key == Key
		    and InEvent.Modifiers == Modifiers;
	}
};


// ─── OaKeyAction ──────────────────────────────────────────────────────────────

struct OaKeyAction {
	OaString       Name;
	OaKeyBinding   Binding;
	OaInputContext Context  = OaInputContext::Global;
	OaFunc<void()> Callback;
};


// ─── OaInputSystem ────────────────────────────────────────────────────────────

class OaInputSystem {
public:
	OaInputSystem() = default;
	OaInputSystem(const OaInputSystem&)            = delete;
	OaInputSystem& operator=(const OaInputSystem&) = delete;
	OaInputSystem(OaInputSystem&&) noexcept        = default;
	OaInputSystem& operator=(OaInputSystem&&) noexcept = default;

	void RegisterAction(OaKeyAction InAction);
	void UnregisterAction(OaStringView InName);
	void SetCallback(OaStringView InName, OaFunc<void()> InCallback);
	void Rebind(OaStringView InName, OaKeyBinding InBinding);

	void SetContext(OaInputContext InCtx) noexcept { Context_ = InCtx; }
	[[nodiscard]] OaInputContext Context() const noexcept { return Context_; }

	// Returns true if InEvent was consumed by a matching action.
	[[nodiscard]] bool Dispatch(const OaUiEvent& InEvent);

	// Register the default Realm Editor bindings:
	//   F12       — screenshot
	//   Ctrl+R    — start/stop recording
	//   Space     — toggle camera
	//   Ctrl+Z    — undo (node graph)
	//   Ctrl+Y    — redo
	//   Ctrl+S    — save
	//   F         — fit-to-view (node canvas)
	void RegisterDefaults();

	[[nodiscard]] OaStatus LoadBindingsYaml(OaStringView InPath);
	[[nodiscard]] OaStatus SaveBindingsYaml(OaStringView InPath) const;

private:
	OaVec<OaKeyAction> Actions_;
	OaInputContext     Context_ = OaInputContext::Global;
};
