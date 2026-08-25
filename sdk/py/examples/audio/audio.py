# OA_DOC_BEGIN: audio-process
import sys

import oa

audio = oa.FnAudio.decodeFile(oa.Paths.asset("audio/oaNarration.wav"))

mono = oa.FnAudio.toMono(audio)

normalized = oa.FnAudio.normalize(mono, -3.0)

faded = oa.FnAudio.fade(normalized, 2400, 2400)

reverberated = oa.FnAudio.reverb(faded, 1.5, 0.4)

output = oa.Paths.var("example/audio/oaNarrationRoom.wav")
oa.Filesystem.createDirectories(output.parentPath())
oa.FnAudio.saveWavF32(output, reverberated)

assert reverberated.isValid()
assert reverberated.sampleRate == 24000
assert reverberated.channelCount == 1
assert reverberated.sampleCount == faded.sampleCount + reverberated.sampleRate * 1500 // 1000
assert oa.Filesystem.isFile(output)

print(f"Saved reverberated audio (1.5 s tail): {output}")

if "--preview" in sys.argv:
	oa.Viewer.preview(output, title="OA audio · room reverb", width=960, height=360)
# OA_DOC_END: audio-process
