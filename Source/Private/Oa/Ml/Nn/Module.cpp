// OaModule — slim implementation (OaModule.md Phase 2)

#include <Oa/Ml/Module.h>
#include <Oa/Ml/Oam.h>
#include <Oa/Ml/Optim.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cstring>

OaMatrix OaModule::Forward(const OaMatrix& InInput) { return InInput; }

void OaModule::Train(OaBool InTraining) {
	Training_ = InTraining;
	for (auto& child : Children_) {
		child.Module->Train(InTraining);
	}
}

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
	InModule->Train(Training_);
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

OaStatus OaModule::SaveWalk(OamModel& OutOam, const OaString& InPrefix) const {
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
			return OaStatus::Error(status.GetCode(),
				"checkpoint readback failed for weight '" + name + "': "
				+ status.GetMessage());
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
			return OaStatus::Error(status.GetCode(),
				"checkpoint readback failed for state '" + name + "': "
				+ status.GetMessage());
		}
		OutOam.AddState(name.c_str(), mat.GetDtype(),
			OaSpan<const OaU64>(shapeVec.Data(), shapeVec.Size()),
			hostBuf.Data(), bytes);
	}
	for (const auto& child : Children_) {
		OA_RETURN_IF_ERROR(child.Module->SaveWalk(
			OutOam, JoinPath(InPrefix, child.Name)));
	}
	return OaStatus::Ok();
}

OaStatus OaModule::SaveTo(OamModel& OutOam) const {
	return SaveWalk(OutOam, OaString());
}

OaStatus OaModule::Save(const OaString& InPath) const {
	OamModel oam;
	OA_RETURN_IF_ERROR(SaveTo(oam));
	return oam.Save(InPath);
}

OaStatus OaModule::Save(const OaString& InPath, const OaOptimizer& InOptimizer) const {
	OamModel oam;
	OA_RETURN_IF_ERROR(SaveTo(oam));
	OA_RETURN_IF_ERROR(InOptimizer.SaveTo(oam));
	return oam.Save(InPath);
}

OaStatus OaModule::ValidateLoadWalk(
	const OamModel& InOam, const OaString& InPrefix) const
{
	for (const auto& p : Params_) {
		const OaString name = JoinPath(InPrefix, p.Name);
		const OamTensorEntry* entry = InOam.FindWeight(name.CStr());
		if (entry == nullptr) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"checkpoint is missing weight '" + name + "'");
		}
		if (entry->Dtype != p.Data.GetDtype()) {
			return OaStatus::Error(OaStatusCode::DtypeMismatch,
				"checkpoint dtype mismatch for weight '" + name + "'");
		}
		if (entry->Rank != static_cast<OaU8>(p.Data.Rank())) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"checkpoint rank mismatch for weight '" + name + "'");
		}
		for (OaI32 dim = 0; dim < p.Data.Rank(); ++dim) {
			if (entry->Shape[dim] != static_cast<OaU64>(p.Data.Size(dim))) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					"checkpoint shape mismatch for weight '" + name + "'");
			}
		}
		if (entry->NumBytes != static_cast<OaU64>(p.Data.ByteSize())) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"checkpoint byte-size mismatch for weight '" + name + "'");
		}
	}
	for (const auto& buffer : Buffers_) {
		if (not buffer.Persistent) continue;
		const OaString name = JoinPath(InPrefix, buffer.Name);
		const OamTensorEntry* entry = InOam.FindState(name.CStr());
		if (entry == nullptr) {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"checkpoint is missing persistent state '" + name + "'");
		}
		if (entry->Dtype != buffer.Data.GetDtype()) {
			return OaStatus::Error(OaStatusCode::DtypeMismatch,
				"checkpoint dtype mismatch for state '" + name + "'");
		}
		if (entry->Rank != static_cast<OaU8>(buffer.Data.Rank())) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"checkpoint rank mismatch for state '" + name + "'");
		}
		for (OaI32 dim = 0; dim < buffer.Data.Rank(); ++dim) {
			if (entry->Shape[dim] != static_cast<OaU64>(buffer.Data.Size(dim))) {
				return OaStatus::Error(OaStatusCode::ShapeMismatch,
					"checkpoint shape mismatch for state '" + name + "'");
			}
		}
		if (entry->NumBytes != static_cast<OaU64>(buffer.Data.ByteSize())) {
			return OaStatus::Error(OaStatusCode::ShapeMismatch,
				"checkpoint byte-size mismatch for state '" + name + "'");
		}
	}
	for (const auto& child : Children_) {
		OA_RETURN_IF_ERROR(child.Module->ValidateLoadWalk(
			InOam, JoinPath(InPrefix, child.Name)));
	}
	return OaStatus::Ok();
}

OaStatus OaModule::LoadWalk(const OamModel& InOam, const OaString& InPrefix) {
	for (auto& p : Params_) {
		OaString name = JoinPath(InPrefix, p.Name);
		const OamTensorEntry* entry = InOam.FindWeight(name.c_str());
		if (entry == nullptr) return OaStatus::Error(OaStatusCode::Internal,
			"validated checkpoint weight disappeared: " + name);
		const void* blobData = InOam.WeightPtr(name.c_str());
		if (blobData == nullptr) return OaStatus::Error(OaStatusCode::Internal,
			"validated checkpoint weight payload disappeared: " + name);
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			const auto status = runtime->UploadBuffer(
				p.Data.GetVkBuffer(), 0, blobData, entry->NumBytes);
			if (not status.IsOk()) return status;
		} else {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"checkpoint restore requires an active OA engine");
		}
	}
	for (auto& buffer : Buffers_) {
		if (!buffer.Persistent) continue;
		OaString name = JoinPath(InPrefix, buffer.Name);
		const OamTensorEntry* entry = InOam.FindState(name.c_str());
		if (entry == nullptr) return OaStatus::Error(OaStatusCode::Internal,
			"validated checkpoint state disappeared: " + name);
		const void* blobData = InOam.StatePtr(name.c_str());
		if (blobData == nullptr) return OaStatus::Error(OaStatusCode::Internal,
			"validated checkpoint state payload disappeared: " + name);
		if (auto* runtime = OaContext::GetDefault().GetEngine()) {
			const auto status = runtime->UploadBuffer(
				buffer.Data.GetVkBuffer(), 0, blobData, entry->NumBytes);
			if (not status.IsOk()) return status;
		} else {
			return OaStatus::Error(OaStatusCode::FailedPrecondition,
				"checkpoint restore requires an active OA engine");
		}
	}
	for (auto& child : Children_) {
		OA_RETURN_IF_ERROR(child.Module->LoadWalk(
			InOam, JoinPath(InPrefix, child.Name)));
	}
	return OaStatus::Ok();
}

OaStatus OaModule::LoadFrom(const OamModel& InOam) {
	OA_RETURN_IF_ERROR(ValidateLoadWalk(InOam, OaString()));
	// Drain any pending context ops so we don't memcpy under in-flight GPU writes.
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Sync());
	return LoadWalk(InOam, OaString());
}

OaStatus OaModule::Load(const OaString& InPath) {
	auto result = OamModel::Load(InPath);
	if (not result.IsOk()) return result.GetStatus();
	auto oam = std::move(result).GetValue();
	return LoadFrom(oam);
}

OaStatus OaModule::Load(const OaString& InPath, OaOptimizer& InOptimizer) {
	auto result = OamModel::Load(InPath);
	if (not result.IsOk()) return result.GetStatus();
	auto oam = std::move(result).GetValue();
	OA_RETURN_IF_ERROR(ValidateLoadWalk(oam, OaString()));
	OA_RETURN_IF_ERROR(InOptimizer.ValidateLoad(oam));
	OA_RETURN_IF_ERROR(LoadFrom(oam));
	return InOptimizer.LoadFrom(oam);
}
