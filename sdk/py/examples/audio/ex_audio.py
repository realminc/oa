# OA_DOC_BEGIN: audio-process
import os
import sys

import oa

engine = oa.Engine()

# Asset path: OA_ASSET env var or the SDK asset directory by convention.
asset_root = os.environ.get("OA_ASSET", "asset")
audio_path = os.path.join(asset_root, "audio/oaNarration.wav")

audio = oa.audio.decode_file(engine, audio_path)

mono = oa.audio.to_mono(audio)

normalized = oa.audio.normalize(mono, -3.0)

faded = oa.audio.fade(normalized, 2400, 2400)

reverberated = oa.audio.reverb(faded, 1.5, 0.4)

var_root = os.environ.get("OA_VAR", "var")
output = os.path.join(var_root, "example/audio/oaNarrationRoom.wav")
os.makedirs(os.path.dirname(output), exist_ok=True)
oa.audio.save_wav_f32(output, reverberated)

assert reverberated.channels == 1
assert os.path.isfile(output)

print(f"Saved reverberated audio (1.5 s tail): {output}")
# OA_DOC_END: audio-process
