// OA Python bindings — stateful audio capture, codec, and playback sessions.
#include "../binding.h"

#include <oa/audio/audioCapture.h>
#include <oa/audio/audioEncoder.h>
#include <oa/audio/audioPlayer.h>

namespace {

template <typename T> T *takeResult(oa::Result<T> &&inResult) {
  if (inResult.isError()) {
    throw std::runtime_error(inResult.getStatus().toString().cStr());
  }
  return new T(oa::move(inResult).getValue());
}

nb::list takePackets(oa::Vec<oa::EncodedAudioPacket> &inPackets) {
  nb::list result;
  for (auto &packet : inPackets) {
    result.append(nb::cast(new oa::EncodedAudioPacket(oa::move(packet)),
                           nb::rv_policy::take_ownership));
  }
  return result;
}

} // namespace

void bindAudioSession(nb::module_ &inModule) {
  nb::enum_<oa::AudioCodec>(inModule, "AudioCodec")
      .value("PcmS16", oa::AudioCodec::PcmS16);

  nb::class_<oa::AudioCaptureConfig>(inModule, "AudioCaptureConfig")
      .def(nb::init<>())
      .def_rw("sampleRate", &oa::AudioCaptureConfig::sampleRate)
      .def_rw("channelCount", &oa::AudioCaptureConfig::channelCount)
      .def_rw("ringMilliseconds", &oa::AudioCaptureConfig::ringMilliseconds);

  nb::class_<oa::AudioCaptureChunk>(inModule, "AudioCaptureChunk")
      .def_prop_ro("interleaved",
                   [](const oa::AudioCaptureChunk &inChunk) {
                     return std::vector<oa::F32>(inChunk.interleaved.begin(),
                                                 inChunk.interleaved.end());
                   })
      .def_ro("sampleRate", &oa::AudioCaptureChunk::sampleRate)
      .def_ro("channelCount", &oa::AudioCaptureChunk::channelCount)
      .def_ro("frameCount", &oa::AudioCaptureChunk::frameCount)
      .def_ro("firstFrameIndex", &oa::AudioCaptureChunk::firstFrameIndex)
      .def_ro("presentationTimestamp",
              &oa::AudioCaptureChunk::presentationTimestamp);

  nb::class_<oa::AudioCapture>(inModule, "AudioCapture")
      .def_static(
          "open",
          [](const oa::AudioCaptureConfig &inConfig) {
            return takeResult(oa::AudioCapture::open(pythonEngine(), inConfig));
          },
          nb::arg("config") = oa::AudioCaptureConfig(),
          nb::rv_policy::take_ownership)
      .def("start",
           [](oa::AudioCapture &inCapture) { throwIfError(inCapture.start()); })
      .def("stop",
           [](oa::AudioCapture &inCapture) { throwIfError(inCapture.stop()); })
      .def(
          "poll",
          [](oa::AudioCapture &inCapture, oa::U32 inMaxFrames) -> nb::object {
            auto *chunk = new oa::AudioCaptureChunk();
            if (not inCapture.poll(*chunk, inMaxFrames)) {
              delete chunk;
              return nb::none();
            }
            return nb::cast(chunk, nb::rv_policy::take_ownership);
          },
          nb::arg("maxFrames") = 4096U)
      .def("close",
           [](oa::AudioCapture &inCapture) { throwIfError(inCapture.close()); })
      .def("isStarted", &oa::AudioCapture::isStarted)
      .def("droppedFrameCount", &oa::AudioCapture::droppedFrameCount);

  nb::class_<oa::AudioEncodeProfile>(inModule, "AudioEncodeProfile")
      .def(nb::init<>())
      .def_rw("codec", &oa::AudioEncodeProfile::codec)
      .def_rw("sampleRate", &oa::AudioEncodeProfile::sampleRate)
      .def_rw("channelCount", &oa::AudioEncodeProfile::channelCount)
      .def_rw("framesPerPacket", &oa::AudioEncodeProfile::framesPerPacket);

  nb::class_<oa::EncodedAudioPacket>(inModule, "EncodedAudioPacket")
      .def_prop_ro("bitstream",
                   [](const oa::EncodedAudioPacket &inPacket) {
                     return nb::bytes(reinterpret_cast<const char *>(
                                          inPacket.bitstream.data()),
                                      inPacket.bitstream.size());
                   })
      .def_ro("presentationFrame", &oa::EncodedAudioPacket::presentationFrame)
      .def_ro("durationFrames", &oa::EncodedAudioPacket::durationFrames);

  nb::class_<oa::AudioEncoder>(inModule, "AudioEncoder")
      .def_static(
          "create",
          [](const oa::AudioEncodeProfile &inProfile) {
            return takeResult(oa::AudioEncoder::create(inProfile));
          },
          nb::arg("profile"), nb::rv_policy::take_ownership)
      .def(
          "encode",
          [](oa::AudioEncoder &inEncoder,
             const std::vector<oa::F32> &inInterleaved) {
            oa::Vec<oa::EncodedAudioPacket> packets;
            throwIfError(
                inEncoder.encode(oa::Span<const oa::F32>(inInterleaved.data(),
                                                         inInterleaved.size()),
                                 packets));
            return takePackets(packets);
          },
          nb::arg("interleaved"))
      .def("flush",
           [](oa::AudioEncoder &inEncoder) {
             oa::Vec<oa::EncodedAudioPacket> packets;
             throwIfError(inEncoder.flush(packets));
             return takePackets(packets);
           })
      .def("close",
           [](oa::AudioEncoder &inEncoder) { throwIfError(inEncoder.close()); })
      .def("profile", &oa::AudioEncoder::getProfile,
           nb::rv_policy::reference_internal)
      .def("codecConfig",
           [](const oa::AudioEncoder &inEncoder) {
             const auto bytes = inEncoder.getCodecConfig();
             return nb::bytes(reinterpret_cast<const char *>(bytes.data()),
                              bytes.size());
           })
      .def("primingFrames", &oa::AudioEncoder::getPrimingFrames)
      .def("isOpen", &oa::AudioEncoder::isOpen);

  nb::class_<oa::AudioPlayerConfig>(inModule, "AudioPlayerConfig")
      .def(nb::init<>())
      .def_prop_rw(
          "uri",
          [](const oa::AudioPlayerConfig &inConfig) {
            return inConfig.uri.stdStr();
          },
          [](oa::AudioPlayerConfig &inConfig, const std::string &inUri) {
            inConfig.uri = oa::String(inUri);
          })
      .def_rw("loop", &oa::AudioPlayerConfig::loop)
      .def_rw("ringMilliseconds", &oa::AudioPlayerConfig::ringMilliseconds);

  nb::class_<oa::AudioPlayer>(inModule, "AudioPlayer")
      .def_static(
          "open",
          [](const oa::AudioPlayerConfig &inConfig) {
            return takeResult(oa::AudioPlayer::open(pythonEngine(), inConfig));
          },
          nb::arg("config"), nb::rv_policy::take_ownership)
      .def("play",
           [](oa::AudioPlayer &inPlayer) { throwIfError(inPlayer.play()); })
      .def("pause", &oa::AudioPlayer::pause)
      .def(
          "seek",
          [](oa::AudioPlayer &inPlayer, oa::U64 inTimestampUs) {
            throwIfError(inPlayer.seek(inTimestampUs));
          },
          nb::arg("timestampUs"))
      .def("setLoop", &oa::AudioPlayer::setLoop, nb::arg("loop"))
      .def("close",
           [](oa::AudioPlayer &inPlayer) { throwIfError(inPlayer.close()); })
      .def("isOpen", &oa::AudioPlayer::isOpen)
      .def("isPlaying", &oa::AudioPlayer::isPlaying)
      .def("isEos", &oa::AudioPlayer::isEos)
      .def("sampleRate", &oa::AudioPlayer::sampleRate)
      .def("channelCount", &oa::AudioPlayer::channelCount)
      .def("durationUs", &oa::AudioPlayer::durationUs)
      .def("positionUs", &oa::AudioPlayer::positionUs)
      .def("underrunFrameCount", &oa::AudioPlayer::underrunFrameCount);
}
