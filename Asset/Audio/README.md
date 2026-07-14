# Audio test assets

`0_jackson_0.wav` is a real human speech recording from the
[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset),
version 1.0.10. It contains the English digit "zero", spoken by Jackson at
8 kHz mono.

The recording is distributed under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).
Source file:
`recordings/0_jackson_0.wav` at commit
`26eb9aaf76e81b692f806f9140c2d2777410d7a1`.
SHA-256: `eea86018ce1730baaf7f5dd6ec88c1f727dd90203521a9115b489310a248ea05`.

`0_jackson_0.flac` is a lossless FLAC transcode of that pinned WAV, produced
with FFmpeg for OA's FLAC decoder gate. It is a ShareAlike derivative under the
same CC BY-SA 4.0 terms. SHA-256:
`c5d2ecd73975347a3c1750151e871453f699a05a968f2ad666aeefd7c7c71bb2`.

`0_jackson_0.mp3` is a 32 kbit/s MP3 transcode used only to gate the built-in
lossy decoder. It is also a CC BY-SA 4.0 derivative. SHA-256:
`c18cd8d6fc3df0edce33e7c89fcb75c3b24037a70019b2f861db9b096f5f12d9`.

The assets are intentionally small and are used only to gate real-file decoding,
GPU feature extraction, processing, WAV-F32 saving, and decode-back validation.
