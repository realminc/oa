# Test video clips

Small, self-contained clips used by `Test/Vision/Codec/TestVideoDecodeReference`
to validate end-to-end `.mp4` decode (container demux → hardware decode →
RGBA readback) against an independent ffmpeg decode, without visual inspection.

Each clip is the **first 2 seconds** (60 frames) of the corresponding
`shibuya_crossing_1080p30_<codec>.mp4` benchmark source, scaled to **720p** and
re-encoded in its native codec so every codec path is exercised with real
inter-frame prediction (I/P/B frames, references) rather than a single keyframe.

| File | Codec | Res | Frames |
|------|-------|-----|--------|
| `shibuya_720p_av1.mp4`  | AV1 (libaom)   | 1280×720 | 60 |
| `shibuya_720p_h264.mp4` | H.264 (x264)   | 1280×720 | 60 |
| `shibuya_720p_h265.mp4` | H.265 (x265)   | 1280×720 | 60 |
| `shibuya_720p_vp9.mp4`  | VP9 (libvpx)   | 1280×720 | 60 |

## Regenerating

From the full benchmark sources (not in-repo — see the dataset folder):

```sh
SRC=/path/to/dataset/video
OUT=Asset/Video
C="-vf scale=1280:720 -t 2 -an -y"
ffmpeg -i $SRC/shibuya_crossing_1080p30_h264.mp4 $C -c:v libx264   -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart $OUT/shibuya_720p_h264.mp4
ffmpeg -i $SRC/shibuya_crossing_1080p30_h265.mp4 $C -c:v libx265   -preset veryfast -crf 25 -tag:v hvc1 -pix_fmt yuv420p -movflags +faststart $OUT/shibuya_720p_h265.mp4
ffmpeg -i $SRC/shibuya_crossing_1080p30_vp9.mp4  $C -c:v libvpx-vp9 -b:v 0 -crf 32 -row-mt 1 -pix_fmt yuv420p $OUT/shibuya_720p_vp9.mp4
ffmpeg -i $SRC/shibuya_crossing_1080p30_av1.mp4  $C -c:v libaom-av1 -cpu-used 8 -crf 35 -b:v 0 -pix_fmt yuv420p -movflags +faststart $OUT/shibuya_720p_av1.mp4
```

To reproduce a decoder bug that only manifests at full resolution, re-cut with
`-c copy` (no scale/re-encode) from the 1080p sources instead.
