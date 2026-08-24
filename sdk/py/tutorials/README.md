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
python sdk/py/tutorials/vision/tutorialVisionBasics.py
```

The package includes checked `oa/*.pyi` files. Pylance therefore resolves
wildcard-imported OA names, completes
`FnMatrix`/`FnImage` operations, and displays their native overloads and
return types.

Re-run the editable install after changing native binding or engine sources.
Pure-Python package and tutorial edits are visible immediately.

The directory hierarchy mirrors the C++ tutorial domains:

```text
py/
  core/
  audio/
  vision/
  plot/
  ml/
    nlp/
```

| Level | Tutorial | Contract proved |
|---|---|---|
| 0 | `core/tutorialCoreBasics.py` | Matrix/scalar operators, in-place semantics, equivalent `FnMatrix` composition, reshape, readback |
| 0 | `audio/tutorialAudioBasics.py` | FLAC decode, `FnAudio` normalize, WAV-F32 save |
| 0 | `vision/tutorialVisionBasics.py` | Format-neutral still-image decode, `FnImage` resize/effect, numerical inspection |
| 1 | `vision/tutorialVisionDataAugmentation.py` | Deterministic GPU augmentation views and headless Plot composition |
| 0 | `plot/tutorialPlotTrainingMetrics.py` | Static training curves rendered to `Image` and displayed through `Viewer` |
| 1 | `ml/tutorialMlBasics.py` | Module, loss, autograd, optimizer, training completion |
| 2 | `ml/tutorialMnistClassifierAutograd.py` | Dataset-backed image classification |
| 3 | `ml/nlp/tutorialNlp*Ag.py` | 16-entry Byte/BPE/Char NLP comparison suite |

Every tutorial source uses only the canonical public root:

```python
from oa import *
```

The controlled NLP matrix is Byte/BPE/Char ×
RNN/GRU/Transformer/MoE/Mamba-3. `TutorialNlpByteEmpyrealmAg.py` is the
sixteenth experimental regression entry. `Ml/Nlp/RunNlpSuite.py` runs the full
matrix or a filtered subset in isolated processes. The old flatten-window
`tutorialNlpRnn.py` and `tutorialNlpRnnAutograd.py` examples were removed
because their names incorrectly implied recurrent-suite coverage.

`vision/tutorialVisionViewer.py` displays a decoded semantic `Image` directly
through `Viewer`. `FnImage.saveFile` provides format-neutral JPEG,
PNG, BMP, TGA, and capability-gated WebP output; its encoded bytes and
round-trip shape/pixel contracts are covered by the Vision codec tests.

`plot/tutorialPlotTrainingMetrics.py` creates deterministic example loss and
accuracy curves, renders the fixed-size figure as a semantic `Image`, and
passes that image through the same `Viewer.show` sink. Static render output
currently omits plot text; the Viewer window title identifies the two panels.
Interactive chart widgets remain outside this tutorial and require a separate
input, selection, and streaming-data contract.
