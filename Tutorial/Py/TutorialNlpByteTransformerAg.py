#!/usr/bin/env python3
"""OA Python Tutorial — Byte-Level Transformer, all-position LM (Autograd).

Python port of Tutorial/Ml/Nlp/TutorialNlpByteTransformerAg.cpp. The attention member of
the NLP fair-comparison suite: a pre-norm transformer
block with causal self-attention, same all-position task / corpus / dims as the
recurrent models.

Model:     token + positional OaEmbedding -> OaTransformerBlock -> OaLayerNorm -> OaLinear
Optimizer: AdamW ; Backward: implicit autograd via OaGradientTape

Run:
    OA_PYTHON_BUILD_DIR=Build/Release PYTHONPATH=Tutorial/Py \
      Build/PythonVenv/bin/python Tutorial/Py/TutorialNlpByteTransformerAg.py
"""

from __future__ import annotations

import array
import math
import os
import tempfile

from _oa_import import core, ml, oa, runtime
import _nlp_common as nlp

VOCAB = nlp.BYTE_VOCAB_SIZE  # 256 — byte vocab family


class ByteTransformerLM:
    """TokEmbed + PosEmbed -> OaTransformerBlock -> LayerNorm (ln_f) -> OaLinear head."""

    def __init__(self):
        wd = core.GetWeightDtype()

        self.tok_embed = ml.OaEmbedding(VOCAB, nlp.D_MODEL)
        self.pos_embed = ml.OaEmbedding(nlp.CONTEXT_LEN, nlp.D_MODEL)
        self.block = ml.OaTransformerBlock(nlp.D_MODEL, nlp.HIDDEN_DIM, nlp.CONTEXT_LEN)
        self.ln_final = ml.OaLayerNorm(nlp.D_MODEL, 1e-5)
        self.head = ml.OaLinear(nlp.D_MODEL, VOCAB)
        self.head.Parameters()[0].Data = core.Rand(VOCAB, nlp.D_MODEL, wd)

        for p in self.parameters():
            p.Data.SetRequiresGrad(True)

    def _position_ids(self, batch):
        n = batch * nlp.CONTEXT_LEN
        ids = array.array("B", (i % nlp.CONTEXT_LEN for i in range(n)))
        return core.FromBytes(ids, n, core.OaScalarType.UInt8)

    def forward(self, x):
        b = x.Size(0)
        n = b * nlp.CONTEXT_LEN
        tok = self.tok_embed.Forward(x).Reshape([n, nlp.D_MODEL])
        pos = self.pos_embed.Forward(self._position_ids(b))
        h = core.Add(tok, pos)
        h = self.block.Forward(h)                       # block reshapes internally via seq_len
        return self.head.Forward(self.ln_final.Forward(h))

    def _mods(self):
        return (("tok_embed", self.tok_embed), ("pos_embed", self.pos_embed),
                ("block", self.block), ("ln_final", self.ln_final), ("head", self.head))

    def parameters(self):
        ps = []
        for _, mod in self._mods():
            ps += list(mod.AllParameterPtrs())
        return ps

    def save(self, prefix):
        for name, mod in self._mods():
            mod.Save(f"{prefix}.{name}.oam")

    def load(self, prefix):
        for name, mod in self._mods():
            mod.Load(f"{prefix}.{name}.oam")


def main():
    runtime.OaInitComputeEngine()
    print("\n" + "=" * 66)
    print("  OA Python Tutorial — Byte Transformer · all-position LM (Autograd)")
    print("=" * 66)
    print(f"Vocab: {VOCAB} bytes · Context: {nlp.CONTEXT_LEN} · DModel: {nlp.D_MODEL} · "
          f"causal self-attention (1 head)")

    model = ByteTransformerLM()
    params = model.parameters()
    n_params = sum(p.Data.NumElements() for p in params)
    print(f"Model: Embed({VOCAB}→{nlp.D_MODEL}) + PosEmbed → "
          f"TransformerBlock({nlp.D_MODEL}, {nlp.HIDDEN_DIM}) → LN → Linear({nlp.D_MODEL}→{VOCAB})")
    print(f"Params: {n_params}    Optimizer: AdamW(lr=0.01)\n")

    sampler = nlp.NlpAllPositionSampler(nlp.CORPUS, nlp.BATCH)
    opt, initial_loss, last_loss, x, y = nlp.train_all_position(
        model.forward, params, sampler, lr=0.01, timer_name="transformer_step")

    acc = nlp.accuracy_all_positions(model.forward, x, y, VOCAB)
    gen = nlp.generate_greedy(model.forward, "hello", 48, VOCAB)
    print(f"\nBaseline: ln({VOCAB}) = {math.log(VOCAB):.4f}")
    print(f"Initial loss: {initial_loss:.4f} · Final loss: {last_loss:.4f} · Accuracy: {acc:.1f}%")
    print(f"Generation:\n  Prompt:    'hello'\n  Generated: '{gen}'")

    prefix = os.path.join(tempfile.gettempdir(), "oa_py_byte_transformer_allpos")
    model.save(prefix)
    reloaded = ByteTransformerLM()
    reloaded.load(prefix)
    reloaded_acc = nlp.accuracy_all_positions(reloaded.forward, x, y, VOCAB)
    print(f"\nSave/Load round-trip: reloaded accuracy {reloaded_acc:.1f}% "
          f"(trained {acc:.1f}%)")

    assert initial_loss > 0.0
    assert last_loss < initial_loss, "Loss must decrease during training"
    assert acc > 30.0, f"All-position accuracy should exceed 30%, got {acc:.1f}%"
    assert abs(reloaded_acc - acc) < 0.5, "Reloaded model must match trained accuracy"
    print("\n✓ All checks passed")


if __name__ == "__main__":
    main()
