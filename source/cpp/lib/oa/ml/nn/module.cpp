// oa::Module — slim implementation (oa::Module.md phase 2)

#include <oa/ml/module.h>
#include <oa/ml/modelFile.h>
#include <oa/ml/optim.h>
#include <oa/core/log.h>
#include <oa/core/matrixAccess.h>
#include <oa/core/std/utility.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>

oa::Matrix oa::Module::forward(const oa::Matrix& inInput) { return inInput; }

void oa::Module::train(oa::Bool inTraining) {
	training_ = inTraining;
	for (auto& child : children_) {
		child.module->train(inTraining);
	}
}

oa::Vector<oa::Parameter> oa::Module::allParameters() {
	oa::Vector<oa::Parameter> all;
	for (auto& p : params_) {
		all.pushBack(p);
	}
	for (auto& child : children_) {
		auto childParams = child.module->allParameters();
		for (auto& cp : childParams) {
			all.pushBack(oa::move(cp));
		}
	}
	return all;
}

oa::Vector<oa::Parameter*> oa::Module::allParameterPtrs() {
	oa::Vector<oa::Parameter*> all;
	for (auto& p : params_) {
		all.pushBack(&p);
	}
	for (auto& child : children_) {
		auto childParams = child.module->allParameterPtrs();
		for (auto* cp : childParams) {
			all.pushBack(cp);
		}
	}
	return all;
}



oa::Vector<oa::ModuleBuffer*> oa::Module::allBufferPtrs(bool inPersistentOnly) {
	oa::Vector<oa::ModuleBuffer*> all;
	for (auto& buffer : buffers_) {
		if (!inPersistentOnly || buffer.persistent) {
			all.pushBack(&buffer);
		}
	}
	for (auto& child : children_) {
		auto childBuffers = child.module->allBufferPtrs(inPersistentOnly);
		for (auto* buffer : childBuffers) {
			all.pushBack(buffer);
		}
	}
	return all;
}

oa::I64 oa::Module::numParameters() const {
	oa::I64 total = 0;
	for (const auto& p : params_) {
		total += p.data.numElements();
	}
	for (const auto& child : children_) {
		total += child.module->numParameters();
	}
	return total;
}

void oa::Module::registerModule(oa::StringView inName, oa::SharedPtr<oa::Module> inModule) {
	inModule->train(training_);
	children_.pushBack({oa::String(inName), oa::move(inModule)});
}

void oa::Module::registerParameter(oa::StringView inName, oa::Matrix inData, bool inRequiresGrad) {
	// Hook the autograd tape: enabling requiresGrad allocates a Tier-1 persistent
	// grad buffer (oaAutograd.md §3) owned by inData's autograd meta — the single
	// source of truth. oa::Parameter::grad() reads it back live, so there is no
	// snapshot to keep in sync (the old gradView copy was the divergence footgun).
	if (inRequiresGrad) {
		inData.setRequiresGrad(true);
	}
	params_.pushBack({oa::String(inName), oa::move(inData), inRequiresGrad});
}

void oa::Module::registerBuffer(oa::StringView inName, oa::Matrix inData, bool inPersistent) {
	inData.setRequiresGrad(false);
	buffers_.pushBack({oa::String(inName), oa::move(inData), inPersistent});
}

// ─── Persistence: non-virtual tree walks ──────────────────────────────────

static oa::String joinPath(const oa::String& inPrefix, const oa::String& inLeaf) {
	return inPrefix.empty() ? inLeaf : (inPrefix + "." + inLeaf);
}

static void namedParameterWalk(
	oa::Module& inModule,
	const oa::String& inPrefix,
	oa::Vector<oa::NamedParameter>& out)
{
	for (auto& p : inModule.parameters()) {
		out.pushBack({joinPath(inPrefix, p.name), &p});
	}
	for (auto& child : inModule.children()) {
		namedParameterWalk(
			*child.module,
			joinPath(inPrefix, child.name),
			out);
	}
}

oa::Vector<oa::NamedParameter> oa::Module::allNamedParameterPtrs() {
	oa::Vector<oa::NamedParameter> all;
	namedParameterWalk(*this, oa::String(), all);
	return all;
}

oa::Status oa::Module::saveWalk(
	oa::Engine& inEngine, ModelFile& outFile,
	const oa::String& inPrefix) const
{
	for (const auto& p : params_) {
		const auto& mat = p.data;
		if (mat.isEmpty()) continue;
		oa::String name = joinPath(inPrefix, p.name);
		oa::Vector<oa::U64> shapeVec;
		for (oa::I32 i = 0; i < mat.rank(); ++i) {
			shapeVec.pushBack(static_cast<oa::U64>(mat.size(i)));
		}
		auto bytes = static_cast<oa::U64>(mat.byteSize());
		oa::Vector<oa::U8> hostBuf(static_cast<oa::I64>(bytes));
		auto status = oa::FnMatrix::copyToHost(mat, hostBuf.data(), bytes);
		if (not status.isOk()) {
			return oa::Status::error(status.getCode(),
				"checkpoint readback failed for weight '" + name + "': "
				+ status.getMessage());
		}
		outFile.addWeight(name.cStr(), mat.getDtype(),
			oa::Span<const oa::U64>(shapeVec.data(), shapeVec.size()),
			hostBuf.data(), bytes);
	}
	for (const auto& buffer : buffers_) {
		if (!buffer.persistent || buffer.data.isEmpty()) continue;
		const auto& mat = buffer.data;
		oa::String name = joinPath(inPrefix, buffer.name);
		oa::Vector<oa::U64> shapeVec;
		for (oa::I32 i = 0; i < mat.rank(); ++i) {
			shapeVec.pushBack(static_cast<oa::U64>(mat.size(i)));
		}
		auto bytes = static_cast<oa::U64>(mat.byteSize());
		oa::Vector<oa::U8> hostBuf(static_cast<oa::I64>(bytes));
		auto status = oa::FnMatrix::copyToHost(mat, hostBuf.data(), bytes);
		if (not status.isOk()) {
			return oa::Status::error(status.getCode(),
				"checkpoint readback failed for state '" + name + "': "
				+ status.getMessage());
		}
		outFile.addState(name.cStr(), mat.getDtype(),
			oa::Span<const oa::U64>(shapeVec.data(), shapeVec.size()),
			hostBuf.data(), bytes);
	}
	for (const auto& child : children_) {
		OA_RETURN_IF_ERROR(child.module->saveWalk(
			inEngine, outFile, joinPath(inPrefix, child.name)));
	}
	return oa::Status::ok();
}

oa::Status oa::Module::saveTo(oa::Engine& inEngine, ModelFile& outFile) const {
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	return saveWalk(inEngine, outFile, oa::String());
}

oa::Status oa::Module::save(
	oa::Engine& inEngine, const oa::String& inPath) const
{
	ModelFile oam;
	OA_RETURN_IF_ERROR(saveTo(inEngine, oam));
	return oam.save(inPath);
}

oa::Status oa::Module::save(
	oa::Engine& inEngine, const oa::String& inPath,
	const oa::Optimizer& inOptimizer) const
{
	ModelFile oam;
	OA_RETURN_IF_ERROR(saveTo(inEngine, oam));
	OA_RETURN_IF_ERROR(inOptimizer.saveTo(inEngine, oam));
	return oam.save(inPath);
}

oa::Status oa::Module::validateLoadWalk(
	const ModelFile& inFile, const oa::String& inPrefix) const
{
	for (const auto& p : params_) {
		const oa::String name = joinPath(inPrefix, p.name);
		const ModelTensorEntry* entry = inFile.findWeight(name.cStr());
		if (entry == nullptr) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"checkpoint is missing weight '" + name + "'");
		}
		if (entry->encoding != ModelTensorEncoding::Dense) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"quantized inference weight cannot load into trainable parameter '" +
				name + "'");
		}
		if (entry->dtype != p.data.getDtype()) {
			return oa::Status::error(oa::StatusCode::DtypeMismatch,
				"checkpoint dtype mismatch for weight '" + name + "'");
		}
		if (entry->rank != static_cast<oa::U8>(p.data.rank())) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"checkpoint rank mismatch for weight '" + name + "'");
		}
		for (oa::I32 dim = 0; dim < p.data.rank(); ++dim) {
			if (entry->shape[dim] != static_cast<oa::U64>(p.data.size(dim))) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					"checkpoint shape mismatch for weight '" + name + "'");
			}
		}
		if (entry->numBytes != static_cast<oa::U64>(p.data.byteSize())) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"checkpoint byte-size mismatch for weight '" + name + "'");
		}
	}
	for (const auto& buffer : buffers_) {
		if (not buffer.persistent) continue;
		const oa::String name = joinPath(inPrefix, buffer.name);
		const ModelTensorEntry* entry = inFile.findState(name.cStr());
		if (entry == nullptr) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"checkpoint is missing persistent state '" + name + "'");
		}
		if (entry->encoding != ModelTensorEncoding::Dense) {
			return oa::Status::error(oa::StatusCode::FailedPrecondition,
				"persistent module state must remain dense: '" + name + "'");
		}
		if (entry->dtype != buffer.data.getDtype()) {
			return oa::Status::error(oa::StatusCode::DtypeMismatch,
				"checkpoint dtype mismatch for state '" + name + "'");
		}
		if (entry->rank != static_cast<oa::U8>(buffer.data.rank())) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"checkpoint rank mismatch for state '" + name + "'");
		}
		for (oa::I32 dim = 0; dim < buffer.data.rank(); ++dim) {
			if (entry->shape[dim] != static_cast<oa::U64>(buffer.data.size(dim))) {
				return oa::Status::error(oa::StatusCode::ShapeMismatch,
					"checkpoint shape mismatch for state '" + name + "'");
			}
		}
		if (entry->numBytes != static_cast<oa::U64>(buffer.data.byteSize())) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"checkpoint byte-size mismatch for state '" + name + "'");
		}
	}
	for (const auto& child : children_) {
		OA_RETURN_IF_ERROR(child.module->validateLoadWalk(
			inFile, joinPath(inPrefix, child.name)));
	}
	return oa::Status::ok();
}

oa::Status oa::Module::loadWalk(
	oa::Engine& inEngine, const ModelFile& inFile,
	const oa::String& inPrefix)
{
	for (auto& p : params_) {
		oa::String name = joinPath(inPrefix, p.name);
		const ModelTensorEntry* entry = inFile.findWeight(name.cStr());
		if (entry == nullptr) return oa::Status::error(oa::StatusCode::Internal,
			"validated checkpoint weight disappeared: " + name);
		const void* blobData = inFile.weightPtr(name.cStr());
		if (blobData == nullptr) return oa::Status::error(oa::StatusCode::Internal,
			"validated checkpoint weight payload disappeared: " + name);
		const auto status = oa::EngineResourceAccess::uploadBuffer(
			inEngine, oa::MatrixAccess::descriptor(p.data),
			0, blobData, entry->numBytes);
		if (not status.isOk()) return status;
	}
	for (auto& buffer : buffers_) {
		if (!buffer.persistent) continue;
		oa::String name = joinPath(inPrefix, buffer.name);
		const ModelTensorEntry* entry = inFile.findState(name.cStr());
		if (entry == nullptr) return oa::Status::error(oa::StatusCode::Internal,
			"validated checkpoint state disappeared: " + name);
		const void* blobData = inFile.statePtr(name.cStr());
		if (blobData == nullptr) return oa::Status::error(oa::StatusCode::Internal,
			"validated checkpoint state payload disappeared: " + name);
		const auto status = oa::EngineResourceAccess::uploadBuffer(
			inEngine, oa::MatrixAccess::descriptor(buffer.data),
			0, blobData, entry->numBytes);
		if (not status.isOk()) return status;
	}
	for (auto& child : children_) {
		OA_RETURN_IF_ERROR(child.module->loadWalk(
			inEngine, inFile, joinPath(inPrefix, child.name)));
	}
	return oa::Status::ok();
}

oa::Status oa::Module::loadFrom(oa::Engine& inEngine, const ModelFile& inFile) {
	OA_RETURN_IF_ERROR(validateLoadWalk(inFile, oa::String()));
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	oa::ExecutionSession::RecordingScope recording(context);
	// drain any pending context ops so we don't memcpy under in-flight GPU writes.
	OA_RETURN_IF_ERROR(context.submitAndWait());
	return loadWalk(inEngine, inFile, oa::String());
}

oa::Status oa::Module::load(oa::Engine& inEngine, const oa::String& inPath) {
	auto result = ModelFile::load(inPath);
	if (not result.isOk()) return result.getStatus();
	auto oam = oa::move(result).getValue();
	return loadFrom(inEngine, oam);
}

oa::Status oa::Module::load(
	oa::Engine& inEngine, const oa::String& inPath,
	oa::Optimizer& inOptimizer)
{
	auto result = ModelFile::load(inPath);
	if (not result.isOk()) return result.getStatus();
	auto oam = oa::move(result).getValue();
	OA_RETURN_IF_ERROR(validateLoadWalk(oam, oa::String()));
	OA_RETURN_IF_ERROR(inOptimizer.validateLoad(oam));
	OA_RETURN_IF_ERROR(loadFrom(inEngine, oam));
	return inOptimizer.loadFrom(inEngine, oam);
}
