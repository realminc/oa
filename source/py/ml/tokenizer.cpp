// OA Python bindings — public text tokenizers.
#include "../binding.h"

#include <oa/ml/tokenizer.h>

#include <string>
#include <vector>

namespace {

std::vector<oa::I32> toStdVector(const oa::Vector<oa::I32>& values) {
	return {values.begin(), values.end()};
}

oa::Vector<oa::I32> toOaVector(const std::vector<oa::I32>& values) {
	return {values.begin(), values.end()};
}

nb::str decodeText(const oa::String& value) {
	PyObject* result = PyUnicode_DecodeUTF8(
		value.cStr(), static_cast<Py_ssize_t>(value.size()), "replace");
	if (result == nullptr) throw nb::python_error();
	return nb::steal<nb::str>(nb::handle(result));
}

} // namespace

void bindMlTokenizer(nb::module_& m) {
	nb::class_<oa::BpeTokenizer>(m, "BpeTokenizer")
		.def(nb::init<oa::I32>(), nb::arg("targetVocab") = 512,
			"Byte-pair tokenizer with a caller-selected vocabulary size")
		.def("train", [](oa::BpeTokenizer& self, const std::string& text,
				oa::I32 numMerges) {
			self.train(text.c_str(), numMerges);
		}, nb::arg("text"), nb::arg("numMerges"))
		.def("encode", [](const oa::BpeTokenizer& self,
				const std::string& text) {
			return toStdVector(self.encode(text.c_str()));
		}, nb::arg("text"))
		.def("decode", [](const oa::BpeTokenizer& self,
				const std::vector<oa::I32>& tokens) {
			return decodeText(self.decode(toOaVector(tokens)));
		}, nb::arg("tokens"))
		.def("save", [](const oa::BpeTokenizer& self, nb::handle path) {
			throwIfError(self.save(pathFromPython(path).string()));
		}, nb::arg("path"))
		.def("load", [](oa::BpeTokenizer& self, nb::handle path) {
			throwIfError(self.load(pathFromPython(path).string()));
		}, nb::arg("path"))
		.def("vocabSize", &oa::BpeTokenizer::vocabSize)
		.def("numMerges", &oa::BpeTokenizer::numMerges);
}
