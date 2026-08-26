#include <oa/ui/input.h>

#include <oa/core/filesystem.h>
#include <oa/core/hostText.h>
#include <oa/core/std/format.h>

#ifdef OA_HAS_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace {

struct KeyName {
	oa::UiKey key;
	const char* name;
};

constexpr KeyName kKeyNames[]{
	KeyName{oa::UiKey::A, "A"}, KeyName{oa::UiKey::B, "B"},
	KeyName{oa::UiKey::C, "C"}, KeyName{oa::UiKey::D, "D"},
	KeyName{oa::UiKey::E, "E"}, KeyName{oa::UiKey::F, "F"},
	KeyName{oa::UiKey::G, "G"}, KeyName{oa::UiKey::H, "H"},
	KeyName{oa::UiKey::I, "I"}, KeyName{oa::UiKey::J, "J"},
	KeyName{oa::UiKey::K, "K"}, KeyName{oa::UiKey::L, "L"},
	KeyName{oa::UiKey::M, "M"}, KeyName{oa::UiKey::N, "N"},
	KeyName{oa::UiKey::O, "O"}, KeyName{oa::UiKey::P, "P"},
	KeyName{oa::UiKey::Q, "Q"}, KeyName{oa::UiKey::R, "R"},
	KeyName{oa::UiKey::S, "S"}, KeyName{oa::UiKey::T, "T"},
	KeyName{oa::UiKey::U, "U"}, KeyName{oa::UiKey::V, "V"},
	KeyName{oa::UiKey::W, "W"}, KeyName{oa::UiKey::X, "X"},
	KeyName{oa::UiKey::Y, "Y"}, KeyName{oa::UiKey::Z, "Z"},
	KeyName{oa::UiKey::Num0, "0"}, KeyName{oa::UiKey::Num1, "1"},
	KeyName{oa::UiKey::Num2, "2"}, KeyName{oa::UiKey::Num3, "3"},
	KeyName{oa::UiKey::Num4, "4"}, KeyName{oa::UiKey::Num5, "5"},
	KeyName{oa::UiKey::Num6, "6"}, KeyName{oa::UiKey::Num7, "7"},
	KeyName{oa::UiKey::Num8, "8"}, KeyName{oa::UiKey::Num9, "9"},
	KeyName{oa::UiKey::Return, "Return"}, KeyName{oa::UiKey::Escape, "Escape"},
	KeyName{oa::UiKey::Backspace, "Backspace"}, KeyName{oa::UiKey::Tab, "Tab"},
	KeyName{oa::UiKey::Space, "Space"}, KeyName{oa::UiKey::Minus, "Minus"},
	KeyName{oa::UiKey::Equals, "Equals"}, KeyName{oa::UiKey::Comma, "Comma"},
	KeyName{oa::UiKey::Period, "Period"}, KeyName{oa::UiKey::Slash, "Slash"},
	KeyName{oa::UiKey::Home, "Home"}, KeyName{oa::UiKey::Delete, "Delete"},
	KeyName{oa::UiKey::End, "End"}, KeyName{oa::UiKey::Right, "Right"},
	KeyName{oa::UiKey::Left, "Left"}, KeyName{oa::UiKey::Down, "Down"},
	KeyName{oa::UiKey::Up, "Up"}, KeyName{oa::UiKey::F1, "F1"},
	KeyName{oa::UiKey::F2, "F2"}, KeyName{oa::UiKey::F3, "F3"},
	KeyName{oa::UiKey::F4, "F4"}, KeyName{oa::UiKey::F5, "F5"},
	KeyName{oa::UiKey::F6, "F6"}, KeyName{oa::UiKey::F7, "F7"},
	KeyName{oa::UiKey::F8, "F8"}, KeyName{oa::UiKey::F9, "F9"},
	KeyName{oa::UiKey::F10, "F10"}, KeyName{oa::UiKey::F11, "F11"},
	KeyName{oa::UiKey::F12, "F12"}, KeyName{oa::UiKey::KpEnter, "KpEnter"},
	KeyName{oa::UiKey::Kp0, "Kp0"}, KeyName{oa::UiKey::Kp1, "Kp1"},
	KeyName{oa::UiKey::Kp2, "Kp2"}, KeyName{oa::UiKey::Kp3, "Kp3"},
	KeyName{oa::UiKey::Kp4, "Kp4"}, KeyName{oa::UiKey::Kp5, "Kp5"},
	KeyName{oa::UiKey::Kp6, "Kp6"}, KeyName{oa::UiKey::Kp7, "Kp7"},
	KeyName{oa::UiKey::Kp8, "Kp8"}, KeyName{oa::UiKey::Kp9, "Kp9"},
};

const char* keyToName(oa::UiKey inKey) noexcept {
	for (const auto& entry : kKeyNames) {
		if (entry.key == inKey) return entry.name;
	}
	return nullptr;
}

bool nameToKey(oa::StringView inName, oa::UiKey& outKey) noexcept {
	for (const auto& entry : kKeyNames) {
		if (inName == entry.name) {
			outKey = entry.key;
			return true;
		}
	}
	return false;
}

const char* contextToName(oa::InputContext inContext) noexcept {
	switch (inContext) {
		case oa::InputContext::Global: return "Global";
		case oa::InputContext::NodeCanvas: return "NodeCanvas";
		case oa::InputContext::TextInput: return "TextInput";
		case oa::InputContext::Timeline: return "Timeline";
	}
	return nullptr;
}

bool nameToContext(
	oa::StringView inName,
	oa::InputContext& outContext) noexcept {
	if (inName == "Global") outContext = oa::InputContext::Global;
	else if (inName == "NodeCanvas") outContext = oa::InputContext::NodeCanvas;
	else if (inName == "TextInput") outContext = oa::InputContext::TextInput;
	else if (inName == "Timeline") outContext = oa::InputContext::Timeline;
	else return false;
	return true;
}

oa::String bindingError(oa::Usize inIndex, oa::StringView inReason) {
	oa::String message("oa::InputSystem binding ");
	message += oa::toString(static_cast<oa::U64>(inIndex));
	message += ": ";
	message += inReason;
	return message;
}

} // namespace

void oa::InputSystem::registerAction(oa::KeyAction inAction) {
	for (auto& a : actions_) {
		if (a.name == inAction.name) {
			a = oa::move(inAction);
			return;
		}
	}
	actions_.pushBack(oa::move(inAction));
}

void oa::InputSystem::unregisterAction(oa::StringView inName) {
	for (oa::U32 i = 0; i < actions_.size(); ++i) {
		if (actions_[i].name == inName) {
			if (i + 1 < actions_.size()) {
				actions_[i] = oa::move(actions_.back());
			}
			actions_.popBack();
			return;
		}
	}
}

void oa::InputSystem::setCallback(oa::StringView inName, oa::Fn<void()> inCallback) {
	for (auto& a : actions_) {
		if (a.name == inName) {
			a.callback = oa::move(inCallback);
			return;
		}
	}
}

void oa::InputSystem::rebind(oa::StringView inName, oa::KeyBinding inBinding) {
	for (auto& a : actions_) {
		if (a.name == inName) {
			a.binding = inBinding;
			return;
		}
	}
}

bool oa::InputSystem::dispatch(const oa::UiEvent& inEvent) {
	if (inEvent.type != oa::UiEventType::KeyDown) return false;
	for (auto& a : actions_) {
		if (a.context != oa::InputContext::Global and a.context != context_) continue;
		if (a.binding.matches(inEvent)) {
			if (inEvent.keyRepeat and not a.allowRepeat) return true;
			if (a.callback) a.callback();
			return true;
		}
	}
	return false;
}

void oa::InputSystem::registerDefaults() {
	registerAction({
		.name = "screenshot",
		.binding = {.key = oa::UiKey::F12},
		.context = oa::InputContext::Global,
		.callback = {},
	});
	registerAction({
		.name = "record",
		.binding = {.key = oa::UiKey::R, .modifiers = oa::UiModifierCtrl},
		.context = oa::InputContext::Global,
		.callback = {},
	});
	registerAction({
		.name = "toggle_camera",
		.binding = {.key = oa::UiKey::Space},
		.context = oa::InputContext::Global,
		.callback = {},
	});
	registerAction({
		.name = "undo",
		.binding = {.key = oa::UiKey::Z, .modifiers = oa::UiModifierCtrl},
		.context = oa::InputContext::NodeCanvas,
		.callback = {},
	});
	registerAction({
		.name = "redo",
		.binding = {.key = oa::UiKey::Y, .modifiers = oa::UiModifierCtrl},
		.context = oa::InputContext::NodeCanvas,
		.callback = {},
	});
	registerAction({
		.name = "save",
		.binding = {.key = oa::UiKey::S, .modifiers = oa::UiModifierCtrl},
		.context = oa::InputContext::Global,
		.callback = {},
	});
	registerAction({
		.name = "fit_view",
		.binding = {.key = oa::UiKey::F},
		.context = oa::InputContext::NodeCanvas,
		.callback = {},
	});
}

oa::Status oa::InputSystem::loadBindingsYaml(oa::StringView inPath) {
	if (inPath.empty()) {
		return oa::Status::invalidArgument(
			"oa::InputSystem::loadBindingsYaml requires a path");
	}
#ifndef OA_HAS_YAML_CPP
	return oa::Status::unimplemented(
		"oa::InputSystem YAML persistence requires yaml-cpp");
#else
	const oa::Path path(inPath);
	if (not oa::Filesystem::isFile(path)) {
		return oa::Status::error(oa::StatusCode::FileNotFound,
			"oa::InputSystem bindings file does not exist: " + path.string());
	}

	struct PendingBinding {
		oa::String name;
		oa::KeyBinding binding;
		oa::InputContext context = oa::InputContext::Global;
		bool allowRepeat = false;
	};
	oa::Vec<PendingBinding> pending;
	try {
		const YAML::Node root = YAML::LoadFile(oa::hostText::copy(path.string()));
		if (not root.IsMap()) {
			return oa::Status::invalidArgument(
				"oa::InputSystem bindings root must be a map");
		}
		if (not root["version"] or not root["version"].IsScalar()
			or root["version"].as<oa::U32>() != 1U) {
			return oa::Status::invalidArgument(
				"oa::InputSystem bindings require version: 1");
		}
		const YAML::Node bindings = root["bindings"];
		if (not bindings or not bindings.IsSequence()) {
			return oa::Status::invalidArgument(
				"oa::InputSystem bindings must be a sequence");
		}

		for (oa::Usize index = 0U; index < bindings.size(); ++index) {
			const YAML::Node node = bindings[index];
			if (not node.IsMap()) {
				return oa::Status::invalidArgument(bindingError(
					index, "entry must be a map"));
			}
			if (not node["action"] or not node["action"].IsScalar()
				or not node["key"] or not node["key"].IsScalar()
				or not node["context"] or not node["context"].IsScalar()) {
				return oa::Status::invalidArgument(bindingError(
					index, "action, key and context are required scalars"));
			}
			PendingBinding item;
			item.name = oa::hostText::copy(node["action"].Scalar());
			if (item.name.empty()) {
				return oa::Status::invalidArgument(bindingError(
					index, "action must not be empty"));
			}
			for (const auto& previous : pending) {
				if (previous.name == item.name) {
					return oa::Status::invalidArgument(bindingError(
						index, "duplicate action"));
				}
			}
			if (not nameToKey(
				oa::hostText::copy(node["key"].Scalar()), item.binding.key)) {
				return oa::Status::invalidArgument(bindingError(
					index, "unknown key"));
			}
			if (not nameToContext(
				oa::hostText::copy(node["context"].Scalar()), item.context)) {
				return oa::Status::invalidArgument(bindingError(
					index, "unknown context"));
			}
			if (node["allow_repeat"]) {
				if (not node["allow_repeat"].IsScalar()) {
					return oa::Status::invalidArgument(bindingError(
						index, "allow_repeat must be a boolean"));
				}
				item.allowRepeat = node["allow_repeat"].as<bool>();
			}
			if (node["modifiers"]) {
				if (not node["modifiers"].IsSequence()) {
					return oa::Status::invalidArgument(bindingError(
						index, "modifiers must be a sequence"));
				}
				for (const auto& modifierNode : node["modifiers"]) {
					if (not modifierNode.IsScalar()) {
						return oa::Status::invalidArgument(bindingError(
							index, "modifier must be a scalar"));
					}
					const oa::String modifier = oa::hostText::copy(modifierNode.Scalar());
					oa::U32 flag = oa::UiModifierNone;
					if (modifier == "shift") flag = oa::UiModifierShift;
					else if (modifier == "ctrl") flag = oa::UiModifierCtrl;
					else if (modifier == "Alt") flag = oa::UiModifierAlt;
					else if (modifier == "Super") flag = oa::UiModifierSuper;
					else {
						return oa::Status::invalidArgument(bindingError(
							index, "unknown modifier"));
					}
					if ((item.binding.modifiers & flag) != 0U) {
						return oa::Status::invalidArgument(bindingError(
							index, "duplicate modifier"));
					}
					item.binding.modifiers |= flag;
				}
			}
			pending.pushBack(oa::move(item));
		}
	} catch (const YAML::Exception& exception) {
		return oa::Status::invalidArgument(
			oa::String("oa::InputSystem invalid bindings YAML: ")
				+ exception.what());
	}

	// apply only after the complete document has been admitted. Existing action
	// callbacks survive a persisted rebind; new actions remain callback-free
	// until their application owner attaches behavior.
	for (auto& item : pending) {
		oa::KeyAction* existing = nullptr;
		for (auto& action : actions_) {
			if (action.name == item.name) {
				existing = &action;
				break;
			}
		}
		if (existing != nullptr) {
			existing->binding = item.binding;
			existing->context = item.context;
			existing->allowRepeat = item.allowRepeat;
		} else {
			registerAction({
				.name = oa::move(item.name),
				.binding = item.binding,
				.context = item.context,
				.allowRepeat = item.allowRepeat,
				.callback = {},
			});
		}
	}
	return oa::Status::ok();
#endif
}

oa::Status oa::InputSystem::saveBindingsYaml(oa::StringView inPath) const {
	if (inPath.empty()) {
		return oa::Status::invalidArgument(
			"oa::InputSystem::saveBindingsYaml requires a path");
	}
#ifndef OA_HAS_YAML_CPP
	return oa::Status::unimplemented(
		"oa::InputSystem YAML persistence requires yaml-cpp");
#else
	YAML::Emitter output;
	output << YAML::BeginMap;
	output << YAML::Key << "version" << YAML::Value << 1;
	output << YAML::Key << "bindings" << YAML::Value << YAML::BeginSeq;
	for (const auto& action : actions_) {
		const char* key = keyToName(action.binding.key);
		const char* context = contextToName(action.context);
		if (action.name.empty() or key == nullptr or context == nullptr
			or (action.binding.modifiers
				& ~(oa::UiModifierShift | oa::UiModifierCtrl | oa::UiModifierAlt | oa::UiModifierSuper)) != 0U) {
			return oa::Status::invalidArgument(
				"oa::InputSystem contains a non-serializable binding");
		}
		output << YAML::BeginMap;
		output << YAML::Key << "action" << YAML::Value << oa::hostText::copy(action.name);
		output << YAML::Key << "key" << YAML::Value << key;
		output << YAML::Key << "modifiers" << YAML::Value << YAML::BeginSeq;
		if ((action.binding.modifiers & oa::UiModifierShift) != 0U) output << "shift";
		if ((action.binding.modifiers & oa::UiModifierCtrl) != 0U) output << "ctrl";
		if ((action.binding.modifiers & oa::UiModifierAlt) != 0U) output << "Alt";
		if ((action.binding.modifiers & oa::UiModifierSuper) != 0U) output << "Super";
		output << YAML::EndSeq;
		output << YAML::Key << "context" << YAML::Value << context;
		output << YAML::Key << "allow_repeat" << YAML::Value
			<< action.allowRepeat;
		output << YAML::EndMap;
	}
	output << YAML::EndSeq << YAML::EndMap;
	if (not output.good()) {
		return oa::Status::error(oa::StatusCode::Internal,
			oa::String("oa::InputSystem failed to encode bindings YAML: ")
				+ oa::hostText::copy(output.GetLastError()));
	}
	return oa::Filesystem::writeText(
		oa::Path(inPath), oa::StringView(output.c_str(), output.size()));
#endif
}
