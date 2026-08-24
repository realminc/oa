// OA Python bindings — ML extensions to the oa::FnMatrix namespace.
#include "../binding.h"

#include <oa/ml/fnMatrix.h>

void bindMlFnMatrix(nb::module_& m) {
	nb::enum_<oa::Quantization>(m, "Quantization")
		.value("Q4", oa::Quantization::Q4)
		.value("Q8", oa::Quantization::Q8);

	nb::class_<oa::QuantMatrix>(m, "QuantMatrix")
		.def(nb::init<>())
		.def("getShape", &oa::QuantMatrix::getShape)
		.def("rank", &oa::QuantMatrix::rank)
		.def("size", &oa::QuantMatrix::size)
		.def("numElements", &oa::QuantMatrix::numElements)
		.def("getQuantization", &oa::QuantMatrix::getQuantization)
		.def("isEmpty", &oa::QuantMatrix::isEmpty);

	m.def("quantize", [](const oa::Matrix& inInput, oa::Quantization inQuantization) {
		return oa::FnMatrix::quantize(inInput, inQuantization);
	}, nb::arg("input"), nb::arg("quantization"));
	m.def("dequantize", [](const oa::QuantMatrix& inInput) {
		return matrixPtr(oa::FnMatrix::dequantize(inInput));
	}, nb::arg("input"));
	m.def("matMulNt", [](
		const oa::Matrix& inInput,
		const oa::QuantMatrix& inWeight)
	{
		return matrixPtr(oa::FnMatrix::matMulNt(inInput, inWeight));
	}, nb::arg("input"), nb::arg("weight"));

	nb::class_<oa::SsmConfig>(m, "SsmConfig")
		.def(nb::init<>())
		.def_rw("batch", &oa::SsmConfig::batch)
		.def_rw("seqLen", &oa::SsmConfig::seqLen)
		.def_rw("nHeads", &oa::SsmConfig::nHeads)
		.def_rw("nGroups", &oa::SsmConfig::nGroups)
		.def_rw("headDim", &oa::SsmConfig::headDim)
		.def_rw("stateSize", &oa::SsmConfig::stateSize)
		.def_rw("numRopeAngles", &oa::SsmConfig::numRopeAngles)
		.def_rw("mimoRank", &oa::SsmConfig::mimoRank)
		.def_rw("hasZ", &oa::SsmConfig::hasZ)
		.def_rw("hasD", &oa::SsmConfig::hasD)
		.def_rw("hasOutNorm", &oa::SsmConfig::hasOutNorm);

	nb::class_<oa::Mamba3PreprocessConfig>(
		m, "Mamba3PreprocessConfig")
		.def(nb::init<>())
		.def_rw("dInner", &oa::Mamba3PreprocessConfig::dInner)
		.def_rw("dState", &oa::Mamba3PreprocessConfig::dState)
		.def_rw("nHeads", &oa::Mamba3PreprocessConfig::nHeads)
		.def_rw("numRopeAngles",
			&oa::Mamba3PreprocessConfig::numRopeAngles)
		.def_rw("nGroups", &oa::Mamba3PreprocessConfig::nGroups)
		.def_rw("mimoRank", &oa::Mamba3PreprocessConfig::mimoRank)
		.def_rw("eps", &oa::Mamba3PreprocessConfig::eps)
		.def_rw("dtMin", &oa::Mamba3PreprocessConfig::dtMin)
		.def_rw("dtMax", &oa::Mamba3PreprocessConfig::dtMax)
		.def_rw("aFloor", &oa::Mamba3PreprocessConfig::aFloor);

	// Schema-v2 operations. The public oa.FnMatrix namespace composes this
	// ML surface with the Core surface without duplicating registrations.
#include "fnMatrixOps.gen.inl"
}
