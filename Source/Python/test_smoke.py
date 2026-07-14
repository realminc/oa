#!/usr/bin/env python3
"""Smoke tests for the OA Python bindings.

Run from a checkout with:

    python -m pytest Source/Python/test_smoke.py

If the native extension is not on PYTHONPATH, set OA_PYTHON_BUILD_DIR to the
directory containing the private `_oa` extension.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_dev_paths() -> None:
    candidates: list[Path] = []

    build_dir = os.getenv("OA_PYTHON_BUILD_DIR")
    if build_dir:
        candidates.append(Path(build_dir).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "Build" / "Release",
            REPO_ROOT / "Build" / "Debug",
            REPO_ROOT / "build",
            REPO_ROOT / "Source" / "Python",
        ]
    )

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_add_dev_paths()

oa = pytest.importorskip("oa", reason="oa Python package is not importable")
core = oa.core
ml = oa.ml
runtime = oa.runtime


@pytest.fixture(scope="session")
def engine():
    if not runtime.OaInitComputeEngine():
        pytest.skip("OA compute engine could not initialize, likely no Vulkan device")
    yield
    shutdown = getattr(runtime, "OaShutdownComputeEngine", None)
    if shutdown is not None:
        shutdown()


def _set_grad_enabled(enabled: bool) -> None:
    setter = getattr(ml, "SetEnabled", None) or getattr(ml, "SetGradEnabled", None)
    if setter is None:
        pytest.skip("autograd enable setter is not bound")
    setter(enabled)


def test_import_surface():
    assert hasattr(runtime, "OaInitComputeEngine")
    assert hasattr(core, "OaMatrixShape")
    assert not hasattr(oa, "OaMatrixShape")
    assert hasattr(oa, "Context")
    assert isinstance(oa.__version__, str)


def test_shape_creation():
    shape = core.OaMatrixShape([3, 4])
    assert shape.Rank == 2
    assert shape.NumElements() == 12
    assert shape[0] == 3
    assert shape[1] == 4


def test_autograd_enable_alias():
    _set_grad_enabled(False)
    if hasattr(ml, "IsEnabled"):
        assert ml.IsEnabled() is False

    _set_grad_enabled(True)
    if hasattr(ml, "IsEnabled"):
        assert ml.IsEnabled() is True

    _set_grad_enabled(False)


def test_matrix_factories_and_readback(engine):
    z = core.Zeros(2, 3)
    o = core.Ones(2, 3)
    f = core.Full(2, 3, 5.0)

    assert z.Rank() == 2
    assert o.NumElements() == 6
    assert f.Size(0) == 2
    assert f.Size(1) == 3

    with oa.Context():
        pass

    assert core.CopyToHost2D(f, 2, 3) == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]


def test_from_floats_and_scale(engine):
    # Float feature upload + Scale — the MNIST input path. A UInt8 matrix would
    # silently produce garbage through Scale/matmul, so float data must come in
    # as Float32 via FromFloats.
    x = core.FromFloats([0.0, 64.0, 128.0, 255.0], 1, 4)
    assert x.Dtype() == core.OaScalarType.Float32
    assert core.CopyToHost(x) == [0.0, 64.0, 128.0, 255.0]

    with oa.Context():
        s = core.Scale(x, 1.0 / 255.0)

    got = core.CopyToHost(s)
    expected = [0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]
    assert all(abs(a - b) < 1e-5 for a, b in zip(got, expected)), got


def test_matmulnt_and_elementwise(engine):
    a = core.Full(2, 3, 1.0)
    b = core.Full(2, 3, 2.0)

    with oa.Context():
        c = core.MatMulNt(a, b)
        add_r = core.Add(a, b)
        sub_r = core.Sub(a, b)
        mul_r = core.Mul(a, b)

    assert core.CopyToHost2D(c, 2, 2) == [[6.0, 6.0], [6.0, 6.0]]
    assert core.CopyToHost2D(add_r, 2, 3) == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    assert core.CopyToHost2D(sub_r, 2, 3) == [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    assert core.CopyToHost2D(mul_r, 2, 3) == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]


def test_gradient_tape_attaches_grad_fn(engine):
    _set_grad_enabled(False)

    x = core.Full(2, 3, 1.0)
    x.SetRequiresGrad(True)

    linear = ml.OaLinear(3, 2)
    for param in linear.Parameters():
        param.Data.SetRequiresGrad(True)

    tape = ml.GradientTape()
    y = linear.Forward(x)

    assert y.HasGradFn()

    del tape
    _set_grad_enabled(False)


def test_recurrent_modules_forward(engine):
    # OaByteEmbedding / OaRnn / OaGru: the NLP-suite building blocks. Verify they
    # construct, expose params, and produce the expected all-position logits shape.
    import array

    b, s, d, h, vocab = 2, 4, 8, 16, 256
    embed = ml.OaByteEmbedding(d)
    rnn = ml.OaRnn(d, h, 1)
    gru = ml.OaGru(d, h, 1)
    head = ml.OaLinear(h, vocab)

    assert len(embed.Parameters()) == 1
    assert len(rnn.AllParameterPtrs()) >= 1
    assert len(gru.AllParameterPtrs()) >= 1

    ids = core.FromBytes(array.array("B", [(i * 7) % 256 for i in range(b * s)]),
                         b, s, core.OaScalarType.UInt8)
    for rec in (rnn, gru):
        e = embed.Forward(ids).Reshape([b, s, d])
        out = rec.Forward(e).Reshape([b * s, h])
        logits = head.Forward(out)
        with oa.Context():
            pass
        assert logits.Shape() == [b * s, vocab]


def test_nlp_suite_modules_construct(engine):
    # The rest of the NLP-suite building blocks: attention + SSM. Verify they
    # construct and expose parameters (training parity is covered by the tutorials).
    d, ff, s, vocab = 8, 16, 4, 256
    ln = ml.OaLayerNorm(d)
    block = ml.OaTransformerBlock(d, ff, s)
    mamba = ml.OaMamba3Module(d, d_state=16, expand=2, head_dim=8, outproj_norm=True)
    empyrealm = ml.OaEmpyrealmCore(vocab, d, d_state=16, expand=2, head_dim=8)

    assert len(ln.AllParameterPtrs()) >= 1
    assert len(block.AllParameterPtrs()) >= 1
    assert len(mamba.AllParameterPtrs()) >= 1
    assert len(empyrealm.AllParameterPtrs()) >= 1


def test_module_save_load_roundtrip(engine, tmp_path):
    # OaModule.Save/Load persist parameters through a file.
    lin = ml.OaLinear(4, 3)
    for p in lin.Parameters():
        p.Data.SetRequiresGrad(True)
    path = str(tmp_path / "linear.oam")
    lin.Save(path)

    reloaded = ml.OaLinear(4, 3)
    reloaded.Load(path)

    x = core.FromFloats([1.0, 2.0, 3.0, 4.0], 1, 4)
    with oa.Context():
        a = lin.Forward(x)
        b = reloaded.Forward(x)
    assert core.CopyToHost(a) == core.CopyToHost(b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
