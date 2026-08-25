// OA Python bindings — neural-network modules.
#include "../binding.h"

#include <oa/ml/nn.h>
#include <oa/ml/byte.h>

void bindMlNn(nb::module_& m) {
    // ═════════════════════════════════════════════════════════════════════════
    // oa::Linear
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Linear, oa::Module>(m, "Linear")
        .def("__init__", [](oa::Linear* self, oa::I32 inFeatures,
                            oa::I32 outFeatures, bool bias) {
            (void)pythonEngine();
            new (self) oa::Linear(inFeatures, outFeatures, bias);
        },
             nb::arg("inFeatures"), nb::arg("outFeatures"), nb::arg("bias") = true,
             "Fully connected linear layer: y = x @ W.t + b")
        .def("forward", [](oa::Linear& self, const oa::Matrix& input) {
            oa::Matrix result = self.forward(input);
            return matrixPtr(std::move(result));
        }, nb::arg("input"), "Forward pass: Out = Input @ W.t + b",
             nb::rv_policy::take_ownership)
        .def("setActivation", &oa::Linear::setActivation, nb::arg("activation"),
             "Set activation function (None, Relu, Gelu)")
        .def("parameters", [](oa::Linear& self) -> std::vector<oa::Parameter*> {
            auto& params = self.parameters();
            std::vector<oa::Parameter*> result;
            result.reserve(params.size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get trainable parameters [weight, bias]");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Embedding
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Embedding, oa::Module>(m, "Embedding")
        .def("__init__", [](oa::Embedding* self, oa::I32 numEmbeddings,
                            oa::I32 embeddingDim) {
            (void)pythonEngine();
            new (self) oa::Embedding(numEmbeddings, embeddingDim);
        },
             nb::arg("numEmbeddings"), nb::arg("embeddingDim"),
             "Embedding lookup layer")
        .def("forward", [](oa::Embedding& self, const oa::Matrix& input) {
            oa::Matrix result = self.forward(input);
            return matrixPtr(std::move(result));
        }, nb::arg("input"), "Forward pass: lookup embeddings for each token index",
             nb::rv_policy::take_ownership)
        .def("parameters", [](oa::Embedding& self) -> std::vector<oa::Parameter*> {
            auto& params = self.parameters();
            std::vector<oa::Parameter*> result;
            result.reserve(params.size());
            for (auto& p : params) result.push_back(&p);
            return result;
        }, nb::rv_policy::reference_internal,
           "Get trainable parameters [embed weight]")
        .def_prop_ro("numEmbeddings", &oa::Embedding::numEmbeddings)
        .def_prop_ro("embeddingDim", &oa::Embedding::embeddingDim);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::ByteEmbedding — byte-vocab (256) embedding; [B, S] ids -> [B, S, d_model].
    // Forward / Parameters / Save / Load are inherited from oa::Module.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::ByteEmbedding, oa::Module>(m, "ByteEmbedding")
        .def(nb::init<oa::I32>(), nb::arg("dModel"),
             "Byte-level embedding (256-symbol vocab, no tokenizer)")
        .def_prop_ro("dModel", &oa::ByteEmbedding::dModel);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Rnn — recurrent tanh cell(s), whole-sequence scan. [B, S, in] -> [B, S, H].
    // Nested module: use allParameterPtrs() (inherited) for the optimizer.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Rnn, oa::Module>(m, "Rnn")
        .def(nb::init<oa::I32, oa::I32, oa::I32, bool>(),
             nb::arg("inputSize"), nb::arg("hiddenSize"),
             nb::arg("numLayers") = 1, nb::arg("bias") = true,
             "Recurrent tanh RNN (fused whole-sequence scan)")
        .def_prop_ro("inputSize", &oa::Rnn::inputSize)
        .def_prop_ro("hiddenSize", &oa::Rnn::hiddenSize)
        .def_prop_ro("numLayers", &oa::Rnn::numLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Gru — gated recurrent unit (reset/update gates), whole-sequence scan.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Gru, oa::Module>(m, "Gru")
        .def(nb::init<oa::I32, oa::I32, oa::I32, bool>(),
             nb::arg("inputSize"), nb::arg("hiddenSize"),
             nb::arg("numLayers") = 1, nb::arg("bias") = true,
             "Gated recurrent unit (fused whole-sequence scan)")
        .def_prop_ro("inputSize", &oa::Gru::inputSize)
        .def_prop_ro("hiddenSize", &oa::Gru::hiddenSize)
        .def_prop_ro("numLayers", &oa::Gru::numLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::LayerNorm — normalization over the last dim. Forward/Parameters inherited.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::LayerNorm, oa::Module>(m, "LayerNorm")
        .def("__init__", [](oa::LayerNorm* self, oa::I32 normalizedShape,
                            oa::F32 eps) {
            (void)pythonEngine();
            new (self) oa::LayerNorm(normalizedShape, eps);
        },
             nb::arg("normalizedShape"), nb::arg("eps") = 1e-5f,
             "Layer normalization over the last dimension");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::TransformerBlock — pre-norm causal self-attention + FFN. Takes flattened
    // [B*S, d_model] and reshapes internally via seqLen.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::TransformerBlock, oa::Module>(m, "TransformerBlock")
        .def("__init__", [](oa::TransformerBlock* self, oa::I32 dModel,
                            oa::I32 dFf, oa::I32 seqLen, oa::F32 eps) {
            (void)pythonEngine();
            new (self) oa::TransformerBlock(dModel, dFf, seqLen, eps);
        },
             nb::arg("dModel"), nb::arg("dFf"), nb::arg("seqLen"), nb::arg("eps") = 1e-5f,
             "Pre-norm one-head transformer block (compatibility constructor)")
        .def("__init__", [](oa::TransformerBlock* self, oa::I32 dModel,
                            oa::I32 dFf, oa::I32 seqLen, oa::I32 numHeads,
                            oa::F32 eps) {
            (void)pythonEngine();
            new (self) oa::TransformerBlock(dModel, dFf, seqLen, numHeads, eps);
        },
             nb::arg("dModel"), nb::arg("dFf"), nb::arg("seqLen"),
             nb::arg("numHeads"), nb::arg("eps") = 1e-5f,
             "Pre-norm multi-head transformer block (causal self-attention + FFN)")
        .def("setSeqLen", &oa::TransformerBlock::setSeqLen, nb::arg("seqLen"),
             "Update the runtime sequence length without replacing model weights");

    nb::class_<oa::NnTransformer, oa::Module>(m, "NnTransformer")
        .def("__init__", [](oa::NnTransformer* self,
                            oa::I32 vocabSize, oa::I32 contextLength,
                            oa::I32 modelWidth, oa::I32 hiddenWidth,
                            oa::I32 numLayers, oa::I32 numHeads, oa::F32 eps) {
            (void)pythonEngine();
            new (self) oa::NnTransformer(
                vocabSize, contextLength, modelWidth, hiddenWidth,
                numLayers, numHeads, eps);
        },
             nb::arg("vocabSize"), nb::arg("contextLength"),
             nb::arg("modelWidth") = 32, nb::arg("hiddenWidth") = 64,
             nb::arg("numLayers") = 1, nb::arg("numHeads") = 1,
             nb::arg("eps") = 1e-5F,
             "Ready-to-train causal Transformer language model")
        .def_prop_ro("vocabSize", &oa::NnTransformer::vocabSize)
        .def_prop_ro("contextLength", &oa::NnTransformer::contextLength)
        .def_prop_ro("modelWidth", &oa::NnTransformer::modelWidth)
        .def_prop_ro("hiddenWidth", &oa::NnTransformer::hiddenWidth)
        .def_prop_ro("numLayers", &oa::NnTransformer::numLayers)
        .def_prop_ro("numHeads", &oa::NnTransformer::numHeads);

    // ═════════════════════════════════════════════════════════════════════════
    // oa::Mamba3Module — Mamba-3 SSM mixer (EXPERIMENTAL). [B, S, D] -> [B, S, D].
    // Full ctor with the reference defaults; the NLP suite overrides d_state /
    // head_dim / outproj_norm.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::Mamba3Module, oa::Module>(m, "Mamba3Module")
        .def(nb::init<oa::I32, oa::I32, oa::I32, oa::I32, oa::I32, oa::F32, bool, oa::I32,
                      oa::F32, oa::F32, oa::F32, oa::F32, bool>(),
             nb::arg("dModel"), nb::arg("dState") = 128, nb::arg("expand") = 2,
             nb::arg("headDim") = 64, nb::arg("nGroups") = 1,
             nb::arg("ropeFraction") = 0.5f, nb::arg("mimo") = false,
             nb::arg("mimoRank") = 4, nb::arg("dtMin") = 0.001f, nb::arg("dtMax") = 0.1f,
             nb::arg("dtInitFloor") = 1e-4f, nb::arg("aFloor") = 1e-4f,
             nb::arg("outprojNorm") = false,
             "Mamba-3 SSM mixer (experimental)");

    // ═════════════════════════════════════════════════════════════════════════
    // oa::EmpyrealmCore — Empyrealm SSM core with an internal embedding
    // (EXPERIMENTAL). [B, S] token ids -> flat [B*S, d_model].
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<oa::EmpyrealmCore, oa::Module>(m, "EmpyrealmCore")
        .def(nb::init<oa::I32, oa::I32, oa::I32, oa::I32, oa::I32, oa::I32, oa::F32, bool, oa::I32,
                      oa::F32, oa::F32, oa::F32, oa::F32, bool>(),
             nb::arg("vocabSize"), nb::arg("dModel"), nb::arg("dState") = 32,
             nb::arg("expand") = 2, nb::arg("headDim") = 16, nb::arg("nGroups") = 1,
             nb::arg("ropeFraction") = 0.5f, nb::arg("mimo") = false,
             nb::arg("mimoRank") = 1, nb::arg("dtMin") = 0.001f, nb::arg("dtMax") = 0.1f,
             nb::arg("dtInitFloor") = 1e-4f, nb::arg("aFloor") = 1e-4f,
             nb::arg("outprojNorm") = true,
             "Empyrealm SSM core with internal embedding (experimental)");
}
