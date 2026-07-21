# OA showcase tutorials — Progressive C++ and Python introduction

**Status:** 📋 DESIGN
**Date:** 2026-07-19
**Module:** Documentation
**Sister docs:** [Submission index](README.md), [Audio API](../../Source/Public/Oa/Audio), [Vision API](../../Source/Public/Oa/Vision), [Tutorial sources](../../Tutorial)

---

## TL;DR

The showcase should teach OA like OpenCV: begin with one useful file-to-output
program, keep early examples under one screen, and add ownership and execution
concepts only when the user needs them. Existing NLP, RL, Vision, and Python
tutorials supply the advanced stages; the missing work is a curated entry path
and complete C++/Python pairs.

---

## Tutorial ladder

| Level | Tutorial | Visible result | Current source | Required follow-up |
|---:|---|---|---|---|
| 0 | Audio file to mel and clean WAV | Output file plus mel shape | Audio codec/operation tests and landing sample | Add paired `TutorialAudioQuickStart.cpp/.py` and one fixture command. |
| 0 | Image load to saved output | Input and output image | `TutorialViewerImageHeadless.cpp` | Reduce to one canonical simple path after the presenter migration. |
| 1 | Matrix multiply | Printed shape/value and explicit completion | `TutorialCoreMatMulIntro.cpp` | Add a matching Python file and installed-package commands. |
| 1 | Vision normalize and resize | Before/after viewer | `TutorialVision*`, `test_vision.py` | Add one paired file-based sample with exact image metadata. |
| 2 | MNIST training | Loss, accuracy, checkpoint | C++ and Python MNIST tutorials | Align names, output, seed, and checkpoint assertions. |
| 2 | CartPole PPO | Live rollout plus before/after return | `TutorialRlCartPolePpo.cpp`, `TutorialRlCartPoleViewer.cpp` | Add Python parity only after the RL binding is complete. |
| 3 | Five NLP architectures | Convergence, generation, reload | `Tutorial/Ml/Nlp`, `Tutorial/Py` | Publish one selector page rather than exposing every binary first. |
| 4 | OaAlm text to motion | Prompt to USD and preview | OaAlm tools and Build Week capture | Keep the early-checkpoint quality label and add an installed-model fixture. |

## Audio quick start

This is the preferred first showcase because it crosses a real codec boundary,
runs useful GPU operations, produces an ML-ready feature matrix, and writes an
artifact. The forms below match the current public API; a shipping tutorial still
needs its own build target and fixture test.

### C++

```cpp
#include <Oa/Audio.h>
#include <Oa/Runtime/Engine.h>

int main() {
    auto engine = OaEngine::Create({});
    if (engine.IsError()) return 1;

    auto decoded = OaAudioDecoder::LoadFile("speech.flac");
    if (decoded.IsError()) return 1;

    auto clean = OaFnAudio::Normalize(decoded->Buffer, -3.0F);
    auto mel = OaFnAudio::MelSpectrogram(clean, decoded->Meta());
    (void)mel;

    auto saved = OaAudioEncoder::SaveWavF32(
        "speech-clean.wav", clean, decoded->SampleRate);
    return saved.IsOk() ? 0 : 1;
}
```

Expected contracts:

- `audio.Buffer`: planar Float32 `[channels, samples]` on the GPU;
- `clean`: same semantic shape;
- `mel`: Float32 `[channels, mel bands, frames]`;
- file decode and WAV encode are explicit synchronous codec boundaries.

### Python

```python
from oa import audio, runtime

assert runtime.OaInitComputeEngine()
decoded = audio.OaAudioDecoder.LoadFile("speech.flac")
clean = audio.Normalize(decoded.Buffer, -3.0, 0)
mel = audio.MelSpectrogram(clean, decoded.SampleRate)
audio.OaAudioEncoder.SaveWavF32(
    "speech-clean.wav", clean, decoded.SampleRate
)
runtime.OaShutdownComputeEngine()
```

The Python example intentionally does not claim “zero Python tax.” It shows that
Python invokes the same native operation names and Vulkan execution library.

### Known sample conflict

The current `Source/Public/Oa/Audio.h` comment and the landing Audio page use
`OaAudioDecoder::LoadFile(...).Unwrap()`. The live `OaResult` contract does not
define `Unwrap()`. The checked form above is canonical until those two stale
samples are corrected. Do not record or publish the landing code block in its
current form.

## API comparison pattern

Each paired tutorial should preserve this mapping:

| Concept | C++ | Python |
|---|---|---|
| Module | `#include <Oa/Audio.h>` | `from oa import audio` |
| Decode | `OaAudioDecoder::LoadFile` | `audio.OaAudioDecoder.LoadFile` |
| Operation | `OaFnAudio::Normalize` | `audio.Normalize` |
| Metadata | `decoded.Meta()` | `decoded.Meta()` |
| Encode | `OaAudioEncoder::SaveWavF32` | `audio.OaAudioEncoder.SaveWavF32` |
| Failure | checked `OaResult` / `OaStatus` | Python exception |

Do not force visual symmetry where ownership differs. C++ should show checked
status at boundaries in the full tutorial, while Python should demonstrate the
exception contract.

## Module walk-through for the website

Use one outcome per module:

| Module | Headline | Demonstration |
|---|---|---|
| Core / Runtime | Allocate, record, submit, inspect. | Matrix multiply with an execution report. |
| ML | Train, evaluate, generate, restore. | Byte Transformer end-to-end gate. |
| RL | Learn while the viewer remains responsive. | CartPole PPO with live return and loss. |
| Vision | Decode, transform, evaluate, display. | Image or video through a two-operation pipeline. |
| Audio | Decode, process, extract, encode. | The quick start above plus mel output. |
| Media | Keep timestamps and completion explicit. | Short decode-transform-record clip. |
| OaAlm | Connect text, tokens, motion, and deployment. | Prompt-to-USD capture labeled as an early proof. |
| Crypto | Batch public-data primitives on the GPU. | Test-oriented diagram; no security marketing demo. |

## Tutorial acceptance contract

A tutorial is publishable only when it has:

1. prerequisites and supported platform/precision;
2. one exact build or installed-package command;
3. sample data with redistributable provenance;
4. a complete source file, not a fragment that hides initialization or completion;
5. expected output with a deterministic or tolerance-based assertion;
6. a failure statement for missing Vulkan capabilities;
7. a focused CI or physical-device gate;
8. a matching screenshot or capture made from the tested binary;
9. C++/Python parity where both surfaces actually ship;
10. no obsolete compatibility-context syntax added after the explicit-execution
    migration.

## Presentation capture order

For Build Week, record only four tutorial outcomes:

1. Audio file to mel spectrogram.
2. Vision transform in the viewer.
3. CartPole PPO learning curve and rollout.
4. NLP generation after checkpoint reload.

The OaAlm video is a fifth optional shot if the final cut remains under 2:55.
