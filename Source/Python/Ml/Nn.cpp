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
             nb::arg("in_features"), nb::arg("out_features"), nb::arg("bias") = true,
             "Fully connected linear layer: y = x @ W.T + b")
        .def("Forward", [](OaLinear& self, const OaMatrix& input) {
            OaMatrix result = self.Forward(input);
            return matrix_ptr(std::move(result));
        }, nb::arg("input"), "Forward pass: Out = Input @ W.T + b",
             nb::rv_policy::take_ownership)
        .def("SetActivation", &OaLinear::SetActivation, nb::arg("activation"),
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
             nb::arg("num_embeddings"), nb::arg("embedding_dim"),
             "Embedding lookup layer")
        .def("Forward", [](OaEmbedding& self, const OaMatrix& input) {
            OaMatrix result = self.Forward(input);
            return matrix_ptr(std::move(result));
        }, nb::arg("input"), "Forward pass: lookup embeddings for each token index",
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
        .def(nb::init<OaI32>(), nb::arg("d_model"),
             "Byte-level embedding (256-symbol vocab, no tokenizer)")
        .def_prop_ro("DModel", &OaByteEmbedding::DModel);

    // ═════════════════════════════════════════════════════════════════════════
    // OaRnn — recurrent tanh cell(s), whole-sequence scan. [B, S, in] -> [B, S, H].
    // Nested module: use AllParameterPtrs() (inherited) for the optimizer.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaRnn, OaModule>(m, "OaRnn")
        .def(nb::init<OaI32, OaI32, OaI32, bool>(),
             nb::arg("input_size"), nb::arg("hidden_size"),
             nb::arg("num_layers") = 1, nb::arg("bias") = true,
             "Recurrent tanh RNN (fused whole-sequence scan)")
        .def_prop_ro("InputSize", &OaRnn::InputSize)
        .def_prop_ro("HiddenSize", &OaRnn::HiddenSize)
        .def_prop_ro("NumLayers", &OaRnn::NumLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // OaGru — gated recurrent unit (reset/update gates), whole-sequence scan.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaGru, OaModule>(m, "OaGru")
        .def(nb::init<OaI32, OaI32, OaI32, bool>(),
             nb::arg("input_size"), nb::arg("hidden_size"),
             nb::arg("num_layers") = 1, nb::arg("bias") = true,
             "Gated recurrent unit (fused whole-sequence scan)")
        .def_prop_ro("InputSize", &OaGru::InputSize)
        .def_prop_ro("HiddenSize", &OaGru::HiddenSize)
        .def_prop_ro("NumLayers", &OaGru::NumLayers);

    // ═════════════════════════════════════════════════════════════════════════
    // OaLayerNorm — normalization over the last dim. Forward/Parameters inherited.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaLayerNorm, OaModule>(m, "OaLayerNorm")
        .def(nb::init<OaI32, OaF32>(),
             nb::arg("normalized_shape"), nb::arg("eps") = 1e-5f,
             "Layer normalization over the last dimension");

    // ═════════════════════════════════════════════════════════════════════════
    // OaTransformerBlock — pre-norm causal self-attention + FFN. Takes flattened
    // [B*S, d_model] and reshapes internally via seq_len.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaTransformerBlock, OaModule>(m, "OaTransformerBlock")
        .def(nb::init<OaI32, OaI32, OaI32, OaF32>(),
             nb::arg("d_model"), nb::arg("d_ff"), nb::arg("seq_len"), nb::arg("eps") = 1e-5f,
             "Pre-norm one-head transformer block (compatibility constructor)")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaF32>(),
             nb::arg("d_model"), nb::arg("d_ff"), nb::arg("seq_len"),
             nb::arg("num_heads"), nb::arg("eps") = 1e-5f,
             "Pre-norm multi-head transformer block (causal self-attention + FFN)")
        .def("SetSeqLen", &OaTransformerBlock::SetSeqLen, nb::arg("seq_len"),
             "Update the runtime sequence length without replacing model weights");

    // ═════════════════════════════════════════════════════════════════════════
    // OaMamba3Module — Mamba-3 SSM mixer (EXPERIMENTAL). [B, S, D] -> [B, S, D].
    // Full ctor with the reference defaults; the NLP suite overrides d_state /
    // head_dim / outproj_norm.
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaMamba3Module, OaModule>(m, "OaMamba3Module")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaI32, OaF32, bool, OaI32,
                      OaF32, OaF32, OaF32, OaF32, bool>(),
             nb::arg("d_model"), nb::arg("d_state") = 128, nb::arg("expand") = 2,
             nb::arg("head_dim") = 64, nb::arg("n_groups") = 1,
             nb::arg("rope_fraction") = 0.5f, nb::arg("mimo") = false,
             nb::arg("mimo_rank") = 4, nb::arg("dt_min") = 0.001f, nb::arg("dt_max") = 0.1f,
             nb::arg("dt_init_floor") = 1e-4f, nb::arg("a_floor") = 1e-4f,
             nb::arg("outproj_norm") = false,
             "Mamba-3 SSM mixer (experimental)");

    // ═════════════════════════════════════════════════════════════════════════
    // OaEmpyrealmCore — Empyrealm SSM core with an internal embedding
    // (EXPERIMENTAL). [B, S] token ids -> flat [B*S, d_model].
    // ═════════════════════════════════════════════════════════════════════════

    nb::class_<OaEmpyrealmCore, OaModule>(m, "OaEmpyrealmCore")
        .def(nb::init<OaI32, OaI32, OaI32, OaI32, OaI32, OaI32, OaF32, bool, OaI32,
                      OaF32, OaF32, OaF32, OaF32, bool>(),
             nb::arg("vocab_size"), nb::arg("d_model"), nb::arg("d_state") = 32,
             nb::arg("expand") = 2, nb::arg("head_dim") = 16, nb::arg("n_groups") = 1,
             nb::arg("rope_fraction") = 0.5f, nb::arg("mimo") = false,
             nb::arg("mimo_rank") = 1, nb::arg("dt_min") = 0.001f, nb::arg("dt_max") = 0.1f,
             nb::arg("dt_init_floor") = 1e-4f, nb::arg("a_floor") = 1e-4f,
             nb::arg("outproj_norm") = true,
             "Empyrealm SSM core with internal embedding (experimental)");
}
