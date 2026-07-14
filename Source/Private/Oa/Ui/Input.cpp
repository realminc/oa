#include <Oa/Ui/Input.h>

void OaInputSystem::RegisterAction(OaKeyAction InAction) {
	for (auto& a : Actions_) {
		if (a.Name == InAction.Name) {
			a = OaStdMove(InAction);
			return;
		}
	}
	Actions_.PushBack(OaStdMove(InAction));
}

void OaInputSystem::UnregisterAction(OaStringView InName) {
	for (OaU32 i = 0; i < Actions_.Size(); ++i) {
		if (Actions_[i].Name == InName) {
			if (i + 1 < Actions_.Size()) {
				Actions_[i] = OaStdMove(Actions_.Back());
			}
			Actions_.PopBack();
			return;
		}
	}
}

void OaInputSystem::SetCallback(OaStringView InName, OaFunc<void()> InCallback) {
	for (auto& a : Actions_) {
		if (a.Name == InName) {
			a.Callback = OaStdMove(InCallback);
			return;
		}
	}
}

void OaInputSystem::Rebind(OaStringView InName, OaKeyBinding InBinding) {
	for (auto& a : Actions_) {
		if (a.Name == InName) {
			a.Binding = InBinding;
			return;
		}
	}
}

bool OaInputSystem::Dispatch(const OaUiEvent& InEvent) {
	if (InEvent.Type != OuiEventType::KeyDown) return false;
	for (auto& a : Actions_) {
		if (a.Context != OaInputContext::Global and a.Context != Context_) continue;
		if (a.Binding.Matches(InEvent)) {
			if (a.Callback) a.Callback();
			return true;
		}
	}
	return false;
}

void OaInputSystem::RegisterDefaults() {}

OaStatus OaInputSystem::LoadBindingsYaml(OaStringView /*InPath*/) {
	return OaStatus::Ok();
}

OaStatus OaInputSystem::SaveBindingsYaml(OaStringView /*InPath*/) const {
	return OaStatus::Ok();
}
