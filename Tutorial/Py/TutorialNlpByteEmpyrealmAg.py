#!/usr/bin/env python3
"""OA Python Tutorial — Byte-Level Empyrealm SSM, all-position LM (Autograd).

Python port of Tutorial/Ml/TutorialNlpByteEmpyrealmAg.cpp. The second state-space
member of the NLP fair-comparison suite: the fused
Empyrealm SSM core (its own internal embedding + mixer + flat residual), same
all-position task / corpus / dims as the rest.

EXPERIMENTAL: the SSM pair is the least-stable surface in the repo (see Release.md).
Trains at lr 0.003 and is kept experimental-labeled for 0.7.

Model:     OaEmpyrealmCore (embed + SSM mixer + residual) -> OaLinear head
Optimizer: AdamW(lr=0.003) ; Backward: implicit autograd via OaGradientTape

Run:
    OA_PYTHON_BUILD_DIR=Build/Release PYTHONPATH=Tutorial/Py \
      Build/PythonVenv/bin/python Tutorial/Py/TutorialNlpByteEmpyrealmAg.py
"""

from __future__ import annotations

import math
import os
import tempfile

from _oa_import import core, ml, oa, runtime
import _nlp_common as nlp

VOCAB = nlp.BYTE_VOCAB_SIZE  # 256 — byte vocab family
D_STATE, EXPAND, HEAD_DIM = 32, 2, 16


class EmpyrealmByteLM:
    """OaEmpyrealmCore (internal embed + SSM mixer + residual) -> OaLinear head."""

    def __init__(self):
        # Core owns the embedding; ctor defaults already match the suite config.
        self.core = ml.OaEmpyrealmCore(
            VOCAB, nlp.D_MODEL, d_state=D_STATE, expand=EXPAND, head_dim=HEAD_DIM,
            n_groups=1, rope_fraction=0.5, mimo=False, mimo_rank=1,
            dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, a_floor=1e-4,
            outproj_norm=True)
        self.head = ml.OaLinear(nlp.D_MODEL, VOCAB)

        for p in self.parameters():
            p.Data.SetRequiresGrad(True)

    def forward(self, x):
        mixed = self.core.Forward(x)      # [B, S] ids -> flat [B*S, D]
        return self.head.Forward(mixed)

    def _mods(self):
        return (("core", self.core), ("head", self.head))

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
    print("  OA Python Tutorial — Byte Empyrealm SSM · all-position LM (Autograd)")
    print("=" * 66)
    print(f"Vocab: {VOCAB} bytes · Context: {nlp.CONTEXT_LEN} · DModel: {nlp.D_MODEL} · "
          f"DState: {D_STATE} (EXPERIMENTAL)")

    model = EmpyrealmByteLM()
    params = model.parameters()
    n_params = sum(p.Data.NumElements() for p in params)
    print(f"Model: EmpyrealmCore(embed + SSM, d_state={D_STATE}) → Linear({nlp.D_MODEL}→{VOCAB})")
    print(f"Params: {n_params}    Optimizer: AdamW(lr=0.003)\n")

    sampler = nlp.NlpAllPositionSampler(nlp.CORPUS, nlp.BATCH)
    opt, initial_loss, last_loss, x, y = nlp.train_all_position(
        model.forward, params, sampler, lr=0.003, timer_name="empyrealm_step")

    acc = nlp.accuracy_all_positions(model.forward, x, y, VOCAB)
    gen = nlp.generate_greedy(model.forward, "hello", 48, VOCAB)
    print(f"\nBaseline: ln({VOCAB}) = {math.log(VOCAB):.4f}")
    print(f"Initial loss: {initial_loss:.4f} · Final loss: {last_loss:.4f} · Accuracy: {acc:.1f}%")
    print(f"Generation:\n  Prompt:    'hello'\n  Generated: '{gen}'")

    prefix = os.path.join(tempfile.gettempdir(), "oa_py_byte_empyrealm_allpos")
    model.save(prefix)
    reloaded = EmpyrealmByteLM()
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
