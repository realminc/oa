# Audio example evidence

Status: **Generated presentation evidence**

`oaNarrationRoom.wav` is copied byte-for-byte from the checked output produced
by `sdk/py/examples/audio/audio.py`. `oaNarrationRoom.mp3` is the browser
fallback encoded from that exact WAV. Regenerate both only after the executable
example and its metadata checks pass:

```bash
.venv/bin/python sdk/py/examples/audio/audio.py
.venv/bin/python tools/documentation/publishAudioExampleEvidence.py
```

The source narration remains `sdk/asset/audio/oaNarration.wav`; the example
inventory binds both source and processed media to the published code.

`oaNarrationRoomViewer.jpg` is a 960×434 crop of the GTK-framed window showing
the same processed WAV through the generated C++ example's optional
`--preview` path. It preserves the native title bar and OA's GPU-rendered
waveform view; it is presentation evidence, not the audio correctness oracle.
The current capture uses the X11 SDL backend only to obtain a deterministic
window image from the GNOME/Wayland session.
