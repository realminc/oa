#include <oa/core/std/assert.h>

#include <oa/ml/byte.h>
#include <oa/ml/autograd/matrix/autogradBlas.h>
#include <oa/core/fnMatrix.h>

oa::Matrix oa::ByteEmbedding::forward(const oa::Matrix& inByteIds) {
	auto& weight = params_[0].data;

	// oa::FnMatrix::gather() now handles both UInt8 and UInt32 on GPU
	OA_REQUIRE_MSG((inByteIds.getDtype() == oa::ScalarType::UInt8 || inByteIds.getDtype() == oa::ScalarType::UInt32), "oa::ByteEmbedding: indices must be UInt8 or UInt32");

	// oa::FnMatrix::gather auto-attaches the embedding-scatter gradient (oa::GradGather)
	// when the table requires grad — no need to hand-wire it here anymore.
	return oa::FnMatrix::gather(weight, inByteIds);
}

oa::Matrix oa::ByteHead::forward(const oa::Matrix& inHidden) {
	auto& weight = params_[0].data;
	auto& bias   = params_[1].data;
	auto out = oa::FnMatrix::linear(inHidden, weight, bias);
	if (oa::FnAutograd::isEnabled() and (inHidden.requiresGrad() or weight.requiresGrad() or bias.requiresGrad())) {
		auto gradFn = oa::makeShared<oa::GradLinear>();
		gradFn->saveForBackward(inHidden, weight);
		gradFn->setGraphInputs(  oa::Vector<oa::Matrix>{inHidden, weight, bias});
		gradFn->sequenceNr_    = oa::FnAutograd::nextSeq();
		gradFn->outputShape_   = out.getShape();  // tape normalizes viewed upstream (e.g. 3D loss reshape) back to [rows,out] before LinearBwd
		out.mutAutograd().gradFn = gradFn;
	}
	return out;
}
