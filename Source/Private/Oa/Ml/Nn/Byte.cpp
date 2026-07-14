#include <cassert>

#include <Oa/Ml/Byte.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Core/FnMatrix.h>

OaMatrix OaByteEmbedding::Forward(const OaMatrix& InByteIds) {
	auto& weight = Params_[0].Data;

	// OaFnMatrix::Gather() now handles both UInt8 and UInt32 on GPU
	assert((InByteIds.GetDtype() == OaScalarType::UInt8 || InByteIds.GetDtype() == OaScalarType::UInt32) &&
		"OaByteEmbedding: indices must be UInt8 or UInt32");

	// OaFnMatrix::Gather auto-attaches the embedding-scatter gradient (OaGradGather)
	// when the table requires grad — no need to hand-wire it here anymore.
	return OaFnMatrix::Gather(weight, InByteIds);
}

OaMatrix OaByteHead::Forward(const OaMatrix& InHidden) {
	auto& weight = Params_[0].Data;
	auto& bias   = Params_[1].Data;
	auto out = OaFnMatrix::Linear(InHidden, weight, bias);
	if (OaFnAutograd::IsEnabled() and (InHidden.RequiresGrad() or weight.RequiresGrad() or bias.RequiresGrad())) {
		auto gradFn = OaMakeSharedPtr<OaGradLinear>();
		gradFn->Saved_         = OaVec<OaMatrix>{InHidden, weight};
		gradFn->SetGraphInputs(  OaVec<OaMatrix>{InHidden, weight, bias});
		gradFn->SequenceNr_    = OaFnAutograd::NextSeq();
		gradFn->OutputShape_   = out.GetShape();  // tape normalizes viewed upstream (e.g. 3D loss reshape) back to [rows,out] before LinearBwd
		out.MutAutograd().GradFn = gradFn;
	}
	return out;
}

