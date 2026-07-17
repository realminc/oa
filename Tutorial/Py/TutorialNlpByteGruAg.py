#!/usr/bin/env python3
"""OA Python Tutorial — Byte-Level GRU, all-position LM (Autograd).

Python port of Tutorial/Ml/Nlp/TutorialNlpByteGruAg.cpp. The gated-recurrent member of
the NLP fair-comparison suite: same all-position
next-token task / corpus / dims as the RNN, swapping OaRnn -> OaGru (reset/update
gates, fused whole-sequence scan).

Model:     OaByteEmbedding -> OaGru -> OaLinear, projected at every step
Optimizer: AdamW ; Backward: implicit autograd via OaGradientTape

Run:
    OA_PYTHON_BUILD_DIR=Build/Release PYTHONPATH=Tutorial/Py \
      Build/PythonVenv/bin/python Tutorial/Py/TutorialNlpByteGruAg.py
"""

from __future__ import annotations

import math
import os
import tempfile

from _oa_import import core, ml, oa, runtime
import _nlp_common as nlp

VOCAB = nlp.BYTE_VOCAB_SIZE  # 256 — byte vocab family


class ByteGruLM:
    """OaByteEmbedding -> OaGru -> OaLinear, projecting every timestep to the vocab."""

    def __init__(self):
        wd = core.GetWeightDtype()

        self.embed = ml.OaByteEmbedding(nlp.D_MODEL)
        self.embed.Parameters()[0].Data = core.RandN(VOCAB, nlp.D_MODEL, wd)

        self.gru = ml.OaGru(nlp.D_MODEL, nlp.HIDDEN_DIM, 1)

        self.head = ml.OaLinear(nlp.HIDDEN_DIM, VOCAB)
        self.head.Parameters()[0].Data = core.Rand(VOCAB, nlp.HIDDEN_DIM, wd)

        for p in self.parameters():
            p.Data.SetRequiresGrad(True)

    def forward(self, x):
        b, s = x.Size(0), x.Size(1)
        e = self.embed.Forward(x).Reshape([b, s, nlp.D_MODEL])
        o = self.gru.Forward(e).Reshape([b * s, nlp.HIDDEN_DIM])
        return self.head.Forward(o)

    def parameters(self):
        return (list(self.embed.Parameters())
                + list(self.gru.AllParameterPtrs())
                + list(self.head.Parameters()))

    def save(self, prefix):
        self.embed.Save(prefix + ".embed.oam")
        self.gru.Save(prefix + ".gru.oam")
        self.head.Save(prefix + ".head.oam")

    def load(self, prefix):
        self.embed.Load(prefix + ".embed.oam")
        self.gru.Load(prefix + ".gru.oam")
        self.head.Load(prefix + ".head.oam")


def main():
    runtime.OaInitComputeEngine()
    print("\n" + "=" * 66)
    print("  OA Python Tutorial — Byte GRU · all-position LM (Autograd)")
    print("=" * 66)
    print(f"Vocab: {VOCAB} bytes · Context: {nlp.CONTEXT_LEN} · "
          f"DModel: {nlp.D_MODEL} · Hidden: {nlp.HIDDEN_DIM}")

    model = ByteGruLM()
    params = model.parameters()
    n_params = sum(p.Data.NumElements() for p in params)
    print(f"Model: ByteEmbed({VOCAB}→{nlp.D_MODEL}) → GRU({nlp.D_MODEL}→{nlp.HIDDEN_DIM}) "
          f"→ Linear({nlp.HIDDEN_DIM}→{VOCAB})")
    print(f"Params: {n_params}    Optimizer: AdamW(lr=0.01)\n")

    sampler = nlp.NlpAllPositionSampler(nlp.CORPUS, nlp.BATCH)
    opt, initial_loss, last_loss, x, y = nlp.train_all_position(
        model.forward, params, sampler, lr=0.01, timer_name="byte_gru_allpos_step")

    acc = nlp.accuracy_all_positions(model.forward, x, y, VOCAB)
    gen = nlp.generate_greedy(model.forward, "hello", 48, VOCAB)
    print(f"\nBaseline: ln({VOCAB}) = {math.log(VOCAB):.4f}")
    print(f"Initial loss: {initial_loss:.4f} · Final loss: {last_loss:.4f} · Accuracy: {acc:.1f}%")
    print(f"Generation:\n  Prompt:    'hello'\n  Generated: '{gen}'")

    prefix = os.path.join(tempfile.gettempdir(), "oa_py_byte_gru_allpos")
    model.save(prefix)
    reloaded = ByteGruLM()
    reloaded.load(prefix)
    reloaded_acc = nlp.accuracy_all_positions(reloaded.forward, x, y, VOCAB)
    print(f"\nSave/Load round-trip: reloaded accuracy {reloaded_acc:.1f}% "
          f"(trained {acc:.1f}%)")

    assert initial_loss > 0.0
    assert last_loss < initial_loss, "Loss must decrease during training"
    assert acc > 50.0, f"All-position accuracy should exceed 50%, got {acc:.1f}%"
    assert abs(reloaded_acc - acc) < 0.5, "Reloaded model must match trained accuracy"
    print("\n✓ All checks passed")


if __name__ == "__main__":
    main()
