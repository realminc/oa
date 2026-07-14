#include <Oa/Core/Node/DeviceDgNode.h>
#include <Oa/Core/CompiledDgGraph.h>
#include <Oa/Core/DeviceDgPlug.h>
#include <Oa/Core/Matrix.h>

void OaDeviceDgNode::DeclareBufferAccess(OaCompiledDgPass, OaBufferAccessDecl&) const {
	// Default: empty declaration. Compiler falls back to global barrier.
}

OaDgPlugDesc OaDeviceDgNode::TensorPlugDesc(OaString InName, const OaMatrix& InTensor) {
	OaDgPlugDesc desc;
	desc.Name = std::move(InName);
	OaMatrix view = InTensor.AsMatrixView();
	desc.Dtype = view.GetDtype();
	desc.Shape = view.GetShape();
	return desc;
}

OaDgPlug* OaDeviceDgNode::AddDeviceTensorInput(OaDgPlugDesc InDesc, OaMatrix* InTensor) {
	InDesc.Dir = OaDgPlugDir::Input;
	auto owned = OaMakeUniquePtr<OaDeviceDgPlug>(this, std::move(InDesc), InTensor);
	OaDgPlug* ptr = owned.Get();
	PlugInputs_.PushBack(ptr);
	PlugsOwned_.PushBack(std::move(owned));
	return ptr;
}

OaDgPlug* OaDeviceDgNode::AddDeviceTensorOutput(OaDgPlugDesc InDesc, OaMatrix* InTensor) {
	InDesc.Dir = OaDgPlugDir::Output;
	auto owned = OaMakeUniquePtr<OaDeviceDgPlug>(this, std::move(InDesc), InTensor);
	OaDgPlug* ptr = owned.Get();
	PlugOutputs_.PushBack(ptr);
	PlugsOwned_.PushBack(std::move(owned));
	return ptr;
}

OaStatus OaDeviceDgNode::Compute(OaDgEvalContext& InCtx) {
	if (!InCtx.HasDeviceRecordBridge()) {
		return OaStatus::Unimplemented(
			OaString("OaDeviceDgNode: CPU Eval requires SetDeviceRecordBridge(engine, batch, backward)")
		);
	}
	OaComputeEngine* rt = InCtx.RecordEngine();
	OaVkBatch* batch = InCtx.RecordBatch();
	if (!rt || !batch) {
		return OaStatus::Error(OaStatusCode::Internal, OaString("OaDeviceDgNode: null record bridge"));
	}
	if (InCtx.DeviceRecordIsBackward()) {
		return RecordBackward(*rt, *batch);
	}
	return RecordForward(*rt, *batch);
}
