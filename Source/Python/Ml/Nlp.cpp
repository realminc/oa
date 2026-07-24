// OA Python bindings — canonical controlled NLP comparison suite.
#include "../Binding.h"

#include <Oa/Ml/Metric.h>
#include <Oa/Ml/NlpSuite.h>

#include <string>
#include <vector>

namespace {

std::vector<OaI32> ToStdVector(const OaVec<OaI32>& values) {
    std::vector<OaI32> result;
    result.reserve(values.Size());
    for (const OaI32 value : values) {
        result.push_back(value);
    }
    return result;
}

OaVec<OaI32> ToOaVector(const std::vector<OaI32>& values) {
    OaVec<OaI32> result(values.size());
    for (OaUsize index = 0; index < values.size(); ++index) {
        result[index] = values[index];
    }
    return result;
}

nb::str DecodeText(const OaString& value) {
    PyObject* result = PyUnicode_DecodeUTF8(
        value.c_str(), static_cast<Py_ssize_t>(value.size()), "replace");
    if (result == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::str>(result);
}

nb::bytes DecodeBytes(const OaString& value) {
    return nb::bytes(value.c_str(), value.size());
}

} // namespace

void BindMlNlp(nb::module_& m) {
    nb::enum_<OaNlpArchitecture>(m, "OaNlpArchitecture")
        .value("Rnn", OaNlpArchitecture::Rnn)
        .value("Gru", OaNlpArchitecture::Gru)
        .value("Transformer", OaNlpArchitecture::Transformer)
        .value("MoeTransformer", OaNlpArchitecture::MoeTransformer)
        .value("Mamba3", OaNlpArchitecture::Mamba3);

    nb::enum_<OaNlpTokenizerKind>(m, "OaNlpTokenizerKind")
        .value("Byte", OaNlpTokenizerKind::Byte)
        .value("Bpe", OaNlpTokenizerKind::Bpe)
        .value("Char", OaNlpTokenizerKind::Char);

    nb::class_<OaNlpSuiteRecipe>(m, "OaNlpSuiteRecipe")
        .def(nb::init<OaNlpArchitecture, OaNlpTokenizerKind>(),
            nb::arg("Architecture") = OaNlpArchitecture::Gru,
            nb::arg("Tokenizer") = OaNlpTokenizerKind::Byte)
        .def("Architecture", &OaNlpSuiteRecipe::Architecture)
        .def("Tokenizer", &OaNlpSuiteRecipe::Tokenizer)
        .def("VocabSize", &OaNlpSuiteRecipe::VocabSize)
        .def("ContextLength", &OaNlpSuiteRecipe::ContextLength)
        .def("ModelWidth", &OaNlpSuiteRecipe::ModelWidth)
        .def("HiddenWidth", &OaNlpSuiteRecipe::HiddenWidth)
        .def("LearningRate", &OaNlpSuiteRecipe::LearningRate)
        .def("ArchitectureId", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.ArchitectureId());
        })
        .def("ArchitectureName", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.ArchitectureName());
        })
        .def("TokenizerId", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.TokenizerId());
        })
        .def("TokenizerName", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.TokenizerName());
        })
        .def("ModelDescription", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.ModelDescription());
        })
        .def("TimerName", [](const OaNlpSuiteRecipe& self) {
            return std::string(self.TimerName());
        });

    nb::class_<OaNlpSuiteModel, OaModule>(m, "OaNlpSuiteModel")
        .def(nb::init<const OaNlpSuiteRecipe&>(), nb::arg("Recipe"))
        .def("SupportsStatefulGeneration",
            &OaNlpSuiteModel::SupportsStatefulGeneration)
        .def("ResetGenerationState", &OaNlpSuiteModel::ResetGenerationState,
            nb::arg("Batch") = 1)
        .def("ForwardGenerationStep",
            [](OaNlpSuiteModel& self, const OaMatrix& token) {
                return matrix_ptr(self.ForwardGenerationStep(token));
            },
            nb::arg("Token"), nb::rv_policy::take_ownership)
        .def("Recipe", &OaNlpSuiteModel::Recipe,
            nb::rv_policy::reference_internal);

    nb::class_<OaNlpSuiteSampler>(m, "OaNlpSuiteSampler")
        .def(nb::init<const OaNlpSuiteRecipe&, OaI32>(),
            nb::arg("Recipe"), nb::arg("BatchSize"))
        .def("Next", [](OaNlpSuiteSampler& self) {
            OaMatrix input;
            OaMatrix target;
            self.Next(input, target);
            return nb::make_tuple(
                nb::cast(matrix_ptr(OaStdMove(input)),
                    nb::rv_policy::take_ownership),
                nb::cast(matrix_ptr(OaStdMove(target)),
                    nb::rv_policy::take_ownership));
        })
        .def("LastSourceUnits", &OaNlpSuiteSampler::LastSourceUnits)
        .def("Encode", [](const OaNlpSuiteSampler& self,
                          const std::string& text) {
            return ToStdVector(self.Encode(text.c_str()));
        }, nb::arg("Text"))
        .def("Decode", [](const OaNlpSuiteSampler& self,
                          const std::vector<OaI32>& tokens) {
            return DecodeText(self.Decode(ToOaVector(tokens)));
        }, nb::arg("Tokens"))
        .def("DecodeBytes", [](const OaNlpSuiteSampler& self,
                               const std::vector<OaI32>& tokens) {
            return DecodeBytes(self.Decode(ToOaVector(tokens)));
        }, nb::arg("Tokens"))
        .def("InputMatrix", [](const OaNlpSuiteSampler& self,
                               const std::vector<OaI32>& tokens) {
            return matrix_ptr(self.InputMatrix(ToOaVector(tokens)));
        }, nb::arg("Tokens"), nb::rv_policy::take_ownership)
        .def("InputStepMatrix", [](const OaNlpSuiteSampler& self,
                                   OaI32 token) {
            return matrix_ptr(self.InputStepMatrix(token));
        }, nb::arg("Token"), nb::rv_policy::take_ownership)
        .def_static("Corpus", [] {
            return std::string(OaNlpSuiteSampler::Corpus());
        });

    m.def("MetricAccuracy", &OaFnMetric::Accuracy,
        nb::arg("Predictions"), nb::arg("Labels"));
    m.def("MetricScalarLoss", &OaFnMetric::ScalarLoss,
        nb::arg("Loss"));

    m.attr("OaNlpSuiteContextLength") = OaNlpSuiteContextLength;
    m.attr("OaNlpSuiteModelWidth") = OaNlpSuiteModelWidth;
    m.attr("OaNlpSuiteHiddenWidth") = OaNlpSuiteHiddenWidth;
    m.attr("OaNlpSuiteTrainingSteps") = OaNlpSuiteTrainingSteps;
    m.attr("OaNlpSuiteBatchSize") = OaNlpSuiteBatchSize;
    m.attr("OaNlpSuiteRngSeed") = OaNlpSuiteRngSeed;
    m.attr("OaNlpSuiteGenerationPrompt") =
        OaNlpSuiteGenerationPrompt;
    m.attr("OaNlpSuiteGenerationSourceUnits") =
        OaNlpSuiteGenerationSourceUnits;
}
