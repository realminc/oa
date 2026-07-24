# OA Python tutorials

Run these against an installed `oapython` wheel or OA's editable development
environment. After installing OA's native dependencies, a source checkout can
be registered with the normal Python packaging command:

```bash
python -m pip install --editable .
```

Select that environment in the editor, then open any tutorial and use **Run
Python File**. A terminal invocation is equally direct:

```bash
python Tutorial/Py/Vision/TutorialVisionBasics.py
```

The package includes checked `oa/*.pyi` files. Pylance therefore resolves
wildcard-imported `Oa*` names, completes
`OaFnMatrix`/`OaFnImage` operations, and displays their native overloads and
return types.

Re-run the editable install after changing native binding or engine sources.
Pure-Python package and tutorial edits are visible immediately.

The directory hierarchy mirrors the C++ tutorial domains:

```text
Py/
  Core/
  Audio/
  Vision/
  Ml/
    Nlp/
```

| Level | Tutorial | Contract proved |
|---|---|---|
| 0 | `Core/TutorialCoreBasics.py` | Matrix/scalar operators, in-place semantics, equivalent `OaFnMatrix` composition, reshape, readback |
| 0 | `Audio/TutorialAudioBasics.py` | FLAC decode, `OaFnAudio` normalize, WAV-F32 save |
| 0 | `Vision/TutorialVisionBasics.py` | Format-neutral still-image decode, `OaFnImage` resize/effect, numerical inspection |
| 1 | `Ml/TutorialMlBasics.py` | Module, loss, autograd, optimizer, training completion |
| 2 | `Ml/TutorialMnistClassifierAutograd.py` | Dataset-backed image classification |
| 3 | `Ml/Nlp/TutorialNlp*Ag.py` | 16-entry Byte/BPE/Char NLP comparison suite |

Every tutorial source uses only the canonical public root:

```python
from oa import *
```

The controlled NLP matrix is Byte/BPE/Char ×
RNN/GRU/Transformer/MoE/Mamba-3. `TutorialNlpByteEmpyrealmAg.py` is the
sixteenth experimental regression entry. `Ml/Nlp/RunNlpSuite.py` runs the full
matrix or a filtered subset in isolated processes. The old flatten-window
`TutorialNlpRnn.py` and `TutorialNlpRnnAutograd.py` examples were removed
because their names incorrectly implied recurrent-suite coverage.

`Vision/TutorialVisionViewer.py` displays a decoded semantic `OaImage` directly
through `OaViewer`. `OaImageEncoder.SaveFile` provides format-neutral JPEG,
PNG, BMP, TGA, and capability-gated WebP output; its encoded bytes and
round-trip shape/pixel contracts are covered by the Vision codec tests.
