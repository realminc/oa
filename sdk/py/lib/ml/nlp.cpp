// OA Python bindings — canonical controlled NLP comparison suite.
#include <binding.h>

#include <ml/nlpSuite.h>

#include <string>
#include <vector>

namespace {

std::vector<oa::I32> toStdVector(const oa::Vector<oa::I32>& values) {
    std::vector<oa::I32> result;
    result.reserve(values.size());
    for (const oa::I32 value : values) {
        result.push_back(value);
    }
    return result;
}

oa::Vector<oa::I32> toOaVector(const std::vector<oa::I32>& values) {
    oa::Vector<oa::I32> result(values.size());
    for (oa::Usize index = 0; index < values.size(); ++index) {
        result[index] = values[index];
    }
    return result;
}

nb::str decodeText(const oa::String& value) {
    PyObject* result = PyUnicode_DecodeUTF8(
        value.cStr(), static_cast<Py_ssize_t>(value.size()), "replace");
    if (result == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::str>(result);
}

nb::bytes decodeBytes(const oa::String& value) {
    return nb::bytes(value.cStr(), value.size());
}

} // namespace

void bindMlNlp(nb::module_& m) {
    nb::enum_<oa::NlpArchitecture>(m, "NlpArchitecture")
        .value("Rnn", oa::NlpArchitecture::Rnn)
        .value("Gru", oa::NlpArchitecture::Gru)
        .value("Transformer", oa::NlpArchitecture::Transformer)
        .value("MoeTransformer", oa::NlpArchitecture::MoeTransformer)
        .value("Mamba3", oa::NlpArchitecture::Mamba3);

    nb::enum_<oa::NlpTokenizerKind>(m, "NlpTokenizerKind")
        .value("Byte", oa::NlpTokenizerKind::Byte)
        .value("Bpe", oa::NlpTokenizerKind::Bpe)
        .value("Char", oa::NlpTokenizerKind::Char);

    nb::class_<oa::NlpSuiteRecipe>(m, "NlpSuiteRecipe")
        .def(nb::init<oa::NlpArchitecture, oa::NlpTokenizerKind>(),
            nb::arg("architecture") = oa::NlpArchitecture::Gru,
            nb::arg("tokenizer") = oa::NlpTokenizerKind::Byte)
        .def("architecture", &oa::NlpSuiteRecipe::architecture)
        .def("tokenizer", &oa::NlpSuiteRecipe::tokenizer)
        .def("vocabSize", &oa::NlpSuiteRecipe::vocabSize)
        .def("contextLength", &oa::NlpSuiteRecipe::contextLength)
        .def("modelWidth", &oa::NlpSuiteRecipe::modelWidth)
        .def("hiddenWidth", &oa::NlpSuiteRecipe::hiddenWidth)
        .def("learningRate", &oa::NlpSuiteRecipe::learningRate)
        .def("architectureId", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.architectureId());
        })
        .def("architectureName", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.architectureName());
        })
        .def("tokenizerId", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.tokenizerId());
        })
        .def("tokenizerName", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.tokenizerName());
        })
        .def("modelDescription", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.modelDescription());
        })
        .def("timerName", [](const oa::NlpSuiteRecipe& self) {
            return std::string(self.timerName());
        });

    nb::class_<oa::NlpSuiteModel, oa::Module>(m, "NlpSuiteModel")
        .def(nb::init<const oa::NlpSuiteRecipe&>(), nb::arg("recipe"))
        .def("supportsStatefulGeneration",
            &oa::NlpSuiteModel::supportsStatefulGeneration)
        .def("resetGenerationState", &oa::NlpSuiteModel::resetGenerationState,
            nb::arg("batch") = 1)
        .def("forwardGenerationStep",
            [](oa::NlpSuiteModel& self, const oa::Matrix& token) {
                return matrixPtr(self.forwardGenerationStep(token));
            },
            nb::arg("token"), nb::rv_policy::take_ownership)
        .def("recipe", &oa::NlpSuiteModel::recipe,
            nb::rv_policy::reference_internal);

    nb::class_<oa::NlpSuiteSampler>(m, "NlpSuiteSampler")
        .def(nb::init<const oa::NlpSuiteRecipe&, oa::I32>(),
            nb::arg("recipe"), nb::arg("batchSize"))
        .def("next", [](oa::NlpSuiteSampler& self) {
            oa::Matrix input;
            oa::Matrix target;
            self.next(input, target);
            return nb::make_tuple(
                nb::cast(matrixPtr(oa::move(input)),
                    nb::rv_policy::take_ownership),
                nb::cast(matrixPtr(oa::move(target)),
                    nb::rv_policy::take_ownership));
        })
        .def("lastSourceUnits", &oa::NlpSuiteSampler::lastSourceUnits)
        .def("encode", [](const oa::NlpSuiteSampler& self,
                          const std::string& text) {
            return toStdVector(self.encode(text.c_str()));
        }, nb::arg("text"))
        .def("decode", [](const oa::NlpSuiteSampler& self, const std::vector<oa::I32>& tokens) {
            return decodeText(self.decode(toOaVector(tokens)));
        }, nb::arg("tokens"))
        .def("decodeBytes", [](const oa::NlpSuiteSampler& self,
                               const std::vector<oa::I32>& tokens) {
            return decodeBytes(self.decode(toOaVector(tokens)));
        }, nb::arg("tokens"))
        .def("inputMatrix", [](const oa::NlpSuiteSampler& self,
                               const std::vector<oa::I32>& tokens) {
            return matrixPtr(self.inputMatrix(toOaVector(tokens)));
        }, nb::arg("tokens"), nb::rv_policy::take_ownership)
        .def("inputStepMatrix", [](const oa::NlpSuiteSampler& self,
                                   oa::I32 token) {
            return matrixPtr(self.inputStepMatrix(token));
        }, nb::arg("token"), nb::rv_policy::take_ownership)
        .def_static("corpus", [] {
            return std::string(oa::NlpSuiteSampler::corpus());
        });

    m.attr("NlpSuiteContextLength") = oa::NlpSuiteContextLength;
    m.attr("NlpSuiteModelWidth") = oa::NlpSuiteModelWidth;
    m.attr("NlpSuiteHiddenWidth") = oa::NlpSuiteHiddenWidth;
    m.attr("NlpSuiteTrainingSteps") = oa::NlpSuiteTrainingSteps;
    m.attr("NlpSuiteBatchSize") = oa::NlpSuiteBatchSize;
    m.attr("NlpSuiteRngSeed") = oa::NlpSuiteRngSeed;
    m.attr("NlpSuiteGenerationPrompt") = oa::NlpSuiteGenerationPrompt;
    m.attr("NlpSuiteGenerationSourceUnits") = oa::NlpSuiteGenerationSourceUnits;
}
