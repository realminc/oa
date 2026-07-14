# OA Mobile Lab

`OaMobileLab` is OA's physical Android training lab. It embeds the real OA Runtime,
Core, ML modules, autograd, optimizer, metrics, checkpoint format, and SPIR-V kernels;
the Android layer owns only lifecycle, driver loading, and presentation.

The app provides:

- app-local Mesa Turnip 26.1.4 through libadrenotools plus an isolated system-driver
  comparison probe;
- the canonical Byte, BPE, and Char tokenizers;
- RNN, GRU, Transformer, sparse MoE Transformer, and Mamba-3 model recipes from
  `OaNlpSuite`;
- live loss/GPU/wall metrics, foreground training, cancellation, generation, and
  resumable `.oam` checkpoints;
- a black-and-white model lab and a dismissible seven-second welcome screen.

## Model support on the reference phone

The current physical gate is Redmi `25062RN2DE` (`creek`), Adreno 610, using Turnip
26.1.4.

| Architecture | Mobile training | Route |
|---|---|---|
| RNN | Verified | Decomposed OA recurrent cell |
| GRU | Verified | Decomposed OA recurrent cell |
| Transformer | Verified | Standard OA attention path |
| MoE Transformer | Verified | Sparse OA experts and shared-memory atomics |
| Mamba-3 | Verified | Bounded mobile backward with global chunk scratch |

Byte, BPE, and Char have each completed a real GRU forward/backward/AdamW/checkpoint
gate. The controlled five-model desktop-versus-phone benchmark is maintained in the
public documentation at [dev.realm.software](https://dev.realm.software/).

## Checkout

libadrenotools is a recursive Git submodule:

```bash
git submodule update --init --recursive \
	Apps/Android/OaMobileLab/third_party/libadrenotools
```

## Build

The source tree stays source-only. Gradle, CMake, and packaged APK outputs use OA's
existing top-level output convention:

```text
Build/Android/OaMobileLab/Gradle/   Gradle intermediates and reports
Build/Android/OaMobileLab/CMake/    NDK/CMake/Ninja intermediates
Build/Android/OaMobileLab/Assets/   verified generated Turnip asset
Bin/Android/OaMobileLab/            installable APKs
```

Build from the app directory:

```bash
cd Apps/Android/OaMobileLab
export JAVA_HOME="$HOME/Apps/android-studio/jbr"
export ANDROID_HOME="$HOME/Android/Sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
./gradlew assembleDebug
```

Install the staged artifact:

```bash
adb install -r ../../../Bin/Android/OaMobileLab/OaMobileLab-debug.apk
```

The build downloads the pinned Turnip archive only when its local asset is absent and
checks both the archive and extracted driver SHA-256 before packaging.

## Run and automate

The UI defaults to bundled Turnip. System and Turnip probes use isolated processes so
one Vulkan loader cannot contaminate the other.

```bash
adb shell am force-stop com.oa.mobilelab.debug
adb shell am start -W \
	-n com.oa.mobilelab.debug/com.oa.mobilelab.MainActivity \
	--es driver turnip
```

The same training entrypoint can run a deterministic smoke gate from ADB:

```bash
adb shell am force-stop com.oa.mobilelab.debug
adb shell am start -S -W --activity-clear-task \
	-n com.oa.mobilelab.debug/com.oa.mobilelab.MainActivity \
	--es architecture transformer --es tokenizer bpe \
	--ez train true --ei steps 1 --ei batch 2
```

Valid architecture IDs are `rnn`, `gru`, `transformer`, `moe`, and `mamba3`. Valid
tokenizer IDs are `byte`, `bpe`, and `char`.

`SMOKE 50 × 8` is only a forward/backward/optimizer/checkpoint compatibility run; it
does not print its random continuation as learned language. `SUITE 300 × 64` is the
quality gate. It matches the desktop tutorial contract: 300 steps,
batch 64, context 16, the canonical 576-byte corpus, prompt `to be`, and 80 generated
source bytes/characters. A one-step ADB run is only a compatibility gate; random output
after one optimizer update is expected and must not be reported as learned language.
Every quality-gate run prints separate `Evaluation`, `Prompt`, and `Generated` fields,
then recreates the model, reloads its checkpoint, and requires generation to match.

Read debug reports without root:

```bash
adb shell run-as com.oa.mobilelab.debug \
	cat files/reports/bpe-transformer-training.txt
adb logcat -s OA hook_impl
```

Run the canonical five-architecture Byte quality matrix on the attached phone:

```bash
Apps/Android/OaMobileLab/tools/run-nlp-suite.sh
```

The runner deletes stale reports/checkpoints, launches every route through the public
activity entrypoint, waits for the foreground service, and rejects cancellation, partial
steps, checkpoint mismatch, or missing fixed-prompt generation. Pass an architecture and
tokenizer to run one route, for example `run-nlp-suite.sh mamba3 byte`. Pass `all` to
run the broader 5-architecture × 3-tokenizer matrix; that is a long compatibility sweep,
not the controlled architecture benchmark reported in the Android NLP suite document.

## Compatibility routes

The reference Turnip compiler cannot compile the 32 KiB-groupshared fused GRU backward
scan, so Android sets `OA_DISABLE_GRU_SCAN=1`; RNN uses its equivalent decomposed route.
These remain GPU-resident OA equations and kernels, not Java or CPU substitutes. The
fused paths remain the default on desktop drivers.

Mamba-3 uses `Mamba3SisoBwdMobile` for the NLP-suite shape. It preserves the verified
backward recurrence, bounds the private state arrays to `N=32`, and uses global chunk
scratch instead of the 32 KiB workgroup history that killed Turnip. Desktop keeps the
faster shared-memory specialization. The physical Adreno 610 completed forward,
backward, AdamW, evaluation, generation, checkpoint save, fresh reload, and deterministic
post-reload generation through this route.

Training runs in an Android `specialUse` foreground service with a persistent progress
notification and an explicit **Stop and checkpoint** action. This is required for the
300-step gate: an ordinary background service was destroyed by the reference phone after
roughly two minutes and therefore could not provide a reliable suite result.
