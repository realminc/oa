// OA Python bindings — neural-network modules.
#include "../Binding.h"

#include <Oa/Ml/Nn.h>
#include <Oa/Ml/Byte.h>

void BindMlNn(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // OaLinear
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaLinear, OaModule>(m, "OaLinear")
        .def(nb::init<OaI32, OaI32, bool>(),
             nb::arg("InFeatures"), nb::arg("OutFeatures"), nb::arg("Bias") = true,
             "Fully connected linear layer: y = x @ W.T + b")
        .def("Forward", [](OaLinear& self, const OaMatrix& input) {
            OaMatrix result = self.Forward(input);
            return matrix_ptr(std::move(result));
        }, nb::arg("Input"), "Forward pass: Out = Input @ W.T + b",
             nb::rv_policy::take_ownership)
        .def("SetActivation", &OaLinear::SetActivation, nb::arg("Activation"),
             "Set activation function (None, Relu, Gelu)")
        .def("Parameters", [](OaLinear& self) -> std::vector<OaParameter*> {
            auto& params = self.Parameters();
            std::vector<OaParameter*> result;
            result.reserve(params.Size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get trainable parameters [weight, bias]");

    // ═════════════════════════════════════════════════════════════════════════
    // OaEmbedding
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaEmbedding, OaModule>(m, "OaEmbedding")
        .def(nb::init<OaI32, OaI32>(),
             nb::arg("NumEmbeddings"), nb::arg("EmbeddingDim"),
             "Embedding lookup layer")
        .def("Forward", [](OaEmbedding& self, const OaMatrix& input) {
            OaMatrix result = self.Forward(input);
            return matrix_ptr(std::move(result));
        }, nb::arg("Input"), "Forward pass: lookup embeddings for each token index",
             nb::rv_policy::take_ownership)
        .def("Parameters", [](OaEmbedding& self) -> std::vector<OaParameter*> {
            auto& params = self.Parameters();
            std::vector<OaParameter*> result;
            result.reserve(params.Size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get trainable parameters [embed weight]")
        .def_prop_ro("NumEmbeddings", &OaEmbedding::NumEmbeddings)
        .def_prop_ro("EmbeddingDim", &OaEmbedding::EmbeddingDim);

    // ═════════════════════════════════════════════════════════════════════════
    // OaByteEmbedding — byte-vocab (256) embedding; [B, S] ids -> [B, S, d_model].
    // Forward / Parameters / Save / Load are inherited from OaModule.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaByteEmbedding, OaModule>(m, "OaByteEmbedding")
        .def(nb::init<OaI32>(), nb::arg("DModel"),
             "Byte-level embedding (256-symbol vocab, no tokenizer)")
        .def_prop_ro("DModel", &OaByteEmbedding::DModel);

    // ═════════════════════════════════════════════════════════════════════════
    // OaRnn — recurrent tanh cell(s), whole-sequence scan. [B, S, in] -> [B, S, H].
    // Nested module: use AllParameterPtrs() (inherited) for the optimizer.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaRnn, OaModule>(m, "OaRnn")
        .def(nb::init<OaI32, OaI32, OaI32, bool>(),
             nb::arg("InputSize"), nb::arg("HiddenSize"),
             nb::arg("NumLayers") = 1, nb::arg("Bias") = true,
             "Recurrent tanh RNN (fused whole-sequence scan)")
        .def_prop_ro("InputSize", &OaRnn::InputSize)
        .def_prop_ro("HiddenSize", &OaRnn::HiddenSize)
        .def_prop_ro("NumLayers", &OaRnn::NumLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // OaGru — gated recurrent unit (reset/update gates), whole-sequence scan.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaGru, OaModule>(m, "OaGru")
        .def(nb::init<OaI32, OaI32, OaI32, bool>(),
             nb::arg("InputSize"), nb::arg("HiddenSize"),
             nb::arg("NumLayers") = 1, nb::arg("Bias") = true,
             "Gated recurrent unit (fused whole-sequence scan)")
        .def_prop_ro("InputSize", &OaGru::InputSize)
        .def_prop_ro("HiddenSize", &OaGru::HiddenSize)
        .def_prop_ro("NumLayers", &OaGru::NumLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // OaLayerNorm — normalization over the last dim. Forward/Parameters inherited.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaLayerNorm, OaModule>(m, "OaLayerNorm")
        .def(nb::init<OaI32, OaF32>(),
             nb::arg("NormalizedShape"), nb::arg("Eps") = 1e-5f,
             "Layer normalization over the last dimension");

    // ═════════════════════════════════════════════════════════════════════════
    // OaTransformerBlock — pre-norm causal self-attention + FFN. Takes flattened
    // [B*S, d_model] and reshapes internally via seq_len.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaTransformerBlock, OaModule>(m, "OaTransformerBlock")
        .def(nb::init<OaI32, OaI32, OaI32, OaF32>(),
             nb::arg("DModel"), nb::arg("DFf"), nb::arg("SeqLen"), nb::arg("Eps") = 1e-5f,
             "Pre-norm one-head transformer block (compatibility constructor)")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaF32>(),
             nb::arg("DModel"), nb::arg("DFf"), nb::arg("SeqLen"),
             nb::arg("NumHeads"), nb::arg("Eps") = 1e-5f,
             "Pre-norm multi-head transformer block (causal self-attention + FFN)")
        .def("SetSeqLen", &OaTransformerBlock::SetSeqLen, nb::arg("SeqLen"),
             "Update the runtime sequence length without replacing model weights");

    // ═════════════════════════════════════════════════════════════════════════
    // OaMamba3Module — Mamba-3 SSM mixer (EXPERIMENTAL). [B, S, D] -> [B, S, D].
    // Full ctor with the reference defaults; the NLP suite overrides d_state /
    // head_dim / outproj_norm.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMamba3Module, OaModule>(m, "OaMamba3Module")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaI32, OaF32, bool, OaI32,
                      OaF32, OaF32, OaF32, OaF32, bool>(),
             nb::arg("DModel"), nb::arg("DState") = 128, nb::arg("Expand") = 2,
             nb::arg("HeadDim") = 64, nb::arg("NGroups") = 1,
             nb::arg("RopeFraction") = 0.5f, nb::arg("Mimo") = false,
             nb::arg("MimoRank") = 4, nb::arg("DtMin") = 0.001f, nb::arg("DtMax") = 0.1f,
             nb::arg("DtInitFloor") = 1e-4f, nb::arg("AFloor") = 1e-4f,
             nb::arg("OutprojNorm") = false,
             "Mamba-3 SSM mixer (experimental)");

    // ═════════════════════════════════════════════════════════════════════════
    // OaEmpyrealmCore — Empyrealm SSM core with an internal embedding
    // (EXPERIMENTAL). [B, S] token ids -> flat [B*S, d_model].
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaEmpyrealmCore, OaModule>(m, "OaEmpyrealmCore")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaI32, OaI32, OaF32, bool, OaI32,
                      OaF32, OaF32, OaF32, OaF32, bool>(),
             nb::arg("VocabSize"), nb::arg("DModel"), nb::arg("DState") = 32,
             nb::arg("Expand") = 2, nb::arg("HeadDim") = 16, nb::arg("NGroups") = 1,
             nb::arg("RopeFraction") = 0.5f, nb::arg("Mimo") = false,
             nb::arg("MimoRank") = 1, nb::arg("DtMin") = 0.001f, nb::arg("DtMax") = 0.1f,
             nb::arg("DtInitFloor") = 1e-4f, nb::arg("AFloor") = 1e-4f,
             nb::arg("OutprojNorm") = true,
             "Empyrealm SSM core with internal embedding (experimental)");
}
