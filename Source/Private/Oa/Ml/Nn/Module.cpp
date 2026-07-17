// OaModule — slim implementation (OaModule.md Phase 2)

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cstring>

OaMatrix OaModule::Forward(const OaMatrix& InInput) { return InInput; }

OaVec<OaParameter> OaModule::AllParameters() {
	OaVec<OaParameter> all;
	for (auto& p : Params_) {
		all.PushBack(p);
	}
	for (auto& child : Children_) {
		auto childParams = child.Module->AllParameters();
		for (auto& cp : childParams) {
			all.PushBack(std::move(cp));
		}
	}
	return all;
}

OaVec<OaParameter*> OaModule::AllParameterPtrs() {
	OaVec<OaParameter*> all;
	for (auto& p : Params_) {
		all.PushBack(&p);
	}
	for (auto& child : Children_) {
		auto childParams = child.Module->AllParameterPtrs();
		for (auto* cp : childParams) {
			all.PushBack(cp);
		}
	}
	return all;
}



OaVec<OaModuleBuffer*> OaModule::AllBufferPtrs(bool InPersistentOnly) {
	OaVec<OaModuleBuffer*> all;
	for (auto& buffer : Buffers_) {
		if (!InPersistentOnly || buffer.Persistent) {
			all.PushBack(&buffer);
		}
	}
	for (auto& child : Children_) {
		auto childBuffers = child.Module->AllBufferPtrs(InPersistentOnly);
		for (auto* buffer : childBuffers) {
			all.PushBack(buffer);
		}
	}
	return all;
}

OaI64 OaModule::NumParameters() const {
	OaI64 total = 0;
	for (const auto& p : Params_) {
		total += p.Data.NumElements();
	}
	for (const auto& child : Children_) {
		total += child.Module->NumParameters();
	}
	return total;
}

void OaModule::RegisterModule(OaStringView InName, OaSharedPtr<OaModule> InModule) {
	Children_.PushBack({OaString(InName), std::move(InModule)});
}

void OaModule::RegisterParameter(OaStringView InName, OaMatrix InData, bool InRequiresGrad) {
	// Hook the autograd tape: enabling RequiresGrad allocates a Tier-1 persistent
	// Grad buffer (OaAutograd.md §3) owned by InData's autograd meta — the single
	// source of truth. OaParameter::Grad() reads it back live, so there is no
	// snapshot to keep in sync (the old gradView copy was the divergence footgun).
	if (InRequiresGrad) {
		InData.SetRequiresGrad(true);
	}
	Params_.PushBack({OaString(InName), std::move(InData), InRequiresGrad});
}

void OaModule::RegisterBuffer(OaStringView InName, OaMatrix InData, bool InPersistent) {
	InData.SetRequiresGrad(false);
	Buffers_.PushBack({OaString(InName), std::move(InData), InPersistent});
}

// ─── Persistence: non-virtual tree walks ──────────────────────────────────

static OaString JoinPath(const OaString& InPrefix, const OaString& InLeaf) {
	return InPrefix.Empty() ? InLeaf : (InPrefix + "." + InLeaf);
}

static void NamedParameterWalk(
	OaModule& InModule,
	const OaString& InPrefix,
	OaVec<OaNamedParameter>& Out)
{
	for (auto& p : InModule.Parameters()) {
		Out.PushBack({JoinPath(InPrefix, p.Name), &p});
	}
	for (auto& child : InModule.Children()) {
		NamedParameterWalk(
			*child.Module,
			JoinPath(InPrefix, child.Name),
			Out);
	}
}

OaVec<OaNamedParameter> OaModule::AllNamedParameterPtrs() {
	OaVec<OaNamedParameter> all;
	NamedParameterWalk(*this, OaString(), all);
	return all;
}

void OaModule::SaveWalk(OamModel& OutOam, const OaString& InPrefix) const {
	for (const auto& p : Params_) {
		const auto& mat = p.Data;
		if (mat.IsEmpty()) continue;
		OaString name = JoinPath(InPrefix, p.Name);
		OaVec<OaU64> shapeVec;
		for (OaI32 i = 0; i < mat.Rank(); ++i) {
			shapeVec.PushBack(static_cast<OaU64>(mat.Size(i)));
		}
		auto bytes = static_cast<OaU64>(mat.ByteSize());
		OaVec<OaU8> hostBuf(static_cast<OaI64>(bytes));
		auto status = OaFnMatrix::CopyToHost(mat, hostBuf.Data(), bytes);
		if (not status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaModule::Save: CopyToHost failed for '%s': %s",
				name.c_str(), status.GetMessage().c_str());
			continue;
		}
		OutOam.AddWeight(name.c_str(), mat.GetDtype(),
			OaSpan<const OaU64>(shapeVec.Data(), shapeVec.Size()),
			hostBuf.Data(), bytes);
	}
	for (const auto& buffer : Buffers_) {
		if (!buffer.Persistent || buffer.Data.IsEmpty()) continue;
		const auto& mat = buffer.Data;
		OaString name = JoinPath(InPrefix, buffer.Name);
		OaVec<OaU64> shapeVec;
		for (OaI32 i = 0; i < mat.Rank(); ++i) {
			shapeVec.PushBack(static_cast<OaU64>(mat.Size(i)));
		}
		auto bytes = static_cast<OaU64>(mat.ByteSize());
		OaVec<OaU8> hostBuf(static_cast<OaI64>(bytes));
		auto status = OaFnMatrix::CopyToHost(mat, hostBuf.Data(), bytes);
		if (not status.IsOk()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaModule::Save: CopyToHost failed for state '%s': %s",
				name.c_str(), status.GetMessage().c_str());
			continue;
		}
		OutOam.AddState(name.c_str(), mat.GetDtype(),
			OaSpan<const OaU64>(shapeVec.Data(), shapeVec.Size()),
			hostBuf.Data(), bytes);
	}
	for (const auto& child : Children_) {
		child.Module->SaveWalk(OutOam, JoinPath(InPrefix, child.Name));
	}
}

void OaModule::SaveTo(OamModel& OutOam) const {
	SaveWalk(OutOam, OaString());
}

OaStatus OaModule::Save(const OaString& InPath) const {
	OamModel oam;
	SaveTo(oam);
	return oam.Save(InPath);
}

OaStatus OaModule::Save(const OaString& InPath, const OaOptimizer& InOptimizer) const {
	OamModel oam;
	SaveTo(oam);
	InOptimizer.SaveTo(oam);
	return oam.Save(InPath);
}

void OaModule::LoadWalk(const OamModel& InOam, const OaString& InPrefix) {
	for (auto& p : Params_) {
		OaString name = JoinPath(InPrefix, p.Name);
		const OamTensorEntry* entry = InOam.FindWeight(name.c_str());
		if (entry == nullptr) {
			OA_LOG_WARN(OaLogComponent::Core,
				"OaModule::Load: weight '%s' not in checkpoint, keeping init value", name.c_str());
			continue;
		}
		const void* blobData = InOam.WeightPtr(name.c_str());
		if (blobData == nullptr) continue;

		const auto expected = static_cast<OaU64>(p.Data.ByteSize());
		if (entry->NumBytes != expected) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaModule::Load: byte size mismatch for '%s' — expected %llu, got %llu",
				name.c_str(),
				static_cast<unsigned long long>(expected),
				static_cast<unsigned long long>(entry->NumBytes));
			continue;
		}
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			const auto status = runtime->UploadBuffer(
				p.Data.GetVkBuffer(), 0, blobData, entry->NumBytes);
			if (!status.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaModule::Load: upload failed for '%s': %s",
					name.c_str(), status.GetMessage().CStr());
			}
		}
	}
	for (auto& buffer : Buffers_) {
		if (!buffer.Persistent) continue;
		OaString name = JoinPath(InPrefix, buffer.Name);
		const OamTensorEntry* entry = InOam.FindState(name.c_str());
		if (entry == nullptr) {
			OA_LOG_WARN(OaLogComponent::Core,
				"OaModule::Load: state '%s' not in checkpoint, keeping init value", name.c_str());
			continue;
		}
		const void* blobData = InOam.StatePtr(name.c_str());
		if (blobData == nullptr) continue;

		const auto expected = static_cast<OaU64>(buffer.Data.ByteSize());
		if (entry->NumBytes != expected || entry->Dtype != buffer.Data.GetDtype()) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"OaModule::Load: state mismatch for '%s'", name.c_str());
			continue;
		}
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			const auto status = runtime->UploadBuffer(
				buffer.Data.GetVkBuffer(), 0, blobData, entry->NumBytes);
			if (!status.IsOk()) {
				OA_LOG_ERROR(OaLogComponent::Core,
					"OaModule::Load: state upload failed for '%s': %s",
					name.c_str(), status.GetMessage().CStr());
			}
		}
	}
	for (auto& child : Children_) {
		child.Module->LoadWalk(InOam, JoinPath(InPrefix, child.Name));
	}
}

void OaModule::LoadFrom(const OamModel& InOam) {
	// Drain any pending context ops so we don't memcpy under in-flight GPU writes.
	(void)OaContext::GetDefault().Execute();
	(void)OaContext::GetDefault().Sync();
	LoadWalk(InOam, OaString());
}

OaStatus OaModule::Load(const OaString& InPath) {
	auto result = OamModel::Load(InPath);
	if (not result.IsOk()) return result.GetStatus();
	auto oam = std::move(result).GetValue();
	LoadFrom(oam);
	return OaStatus::Ok();
}

OaStatus OaModule::Load(const OaString& InPath, OaOptimizer& InOptimizer) {
	auto result = OamModel::Load(InPath);
	if (not result.IsOk()) return result.GetStatus();
	auto oam = std::move(result).GetValue();
	LoadFrom(oam);
	InOptimizer.LoadFrom(oam);
	return OaStatus::Ok();
}
