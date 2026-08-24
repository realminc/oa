#include <oa/vision/videoEncoder.h>

#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/texture.h>

#include "../../../runtime/textureAccess.h"
#include "videoEncoderInternal.h"

namespace {

oa::Status encodeRecorded(
	oa::ExecutionSession& inContext,
	oa::VideoEncoder& inSession,
	const oa::Texture& inRgba,
	oa::EncodedVideoPacket& outFrame,
	oa::U64 inPts)
{
#ifdef OA_ANDROID_ML
	(void)inContext;
	(void)inSession;
	(void)inRgba;
	(void)outFrame;
	(void)inPts;
	return oa::Status::unimplemented(
		"Video encoding is not part of the android ML profile");
#else
	if (not inRgba.isValid()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::encode: source texture is invalid");
	}
	if (inRgba.isImageBacked()) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::encode: source texture must be buffer-backed");
	}
	if (inRgba.width() <= 0 or inRgba.height() <= 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::encode: source texture extent must be positive");
	}
	const oavk::Buffer* buffer = oa::TextureAccess::buffer(inRgba);
	oa::Engine& engine = inContext.engine();
	const oa::U64 bytes = static_cast<oa::U64>(inRgba.width())
		* static_cast<oa::U64>(inRgba.height()) * 4U;
	if (buffer == nullptr or buffer->buffer == nullptr or buffer->size < bytes
		or buffer->bindlessIndex == OA_BINDLESS_INVALID
		or buffer->allocation == nullptr
		or buffer->aliasIdentity != nullptr
		or oa::TextureAccess::engine(inRgba) != &engine
		or buffer->allocatorIdentity != oa::EngineAllocatorAccess::get(engine).allocator) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::VideoEncoder::encode: source must be a non-aliased buffer owned by the context engine");
	}
	oa::ExecutionSession::RecordingScope recording(inContext);
	// encoding is an explicit consumer boundary. submit and complete any graph
	// work that produces the texture before the encoder snapshots its buffer.
	OA_RETURN_IF_ERROR(inContext.submitAndWait());
	OA_RETURN_IF_ERROR(oa::VideoEncoderAccess::uploadInputRgba(inSession,
		*buffer,
		static_cast<oa::U32>(inRgba.width()),
		static_cast<oa::U32>(inRgba.height()),
		oa::YCbCrModel::BT709,
		false));
	return oa::VideoEncoderAccess::encodeFrame(
		inSession, VK_NULL_HANDLE, inPts, outFrame);
#endif
}

} // namespace

oa::Status oa::VideoEncoder::encode(
	const oa::Texture& inRgba,
	oa::EncodedVideoPacket& outFrame,
	oa::U64 inPts)
{
	auto* engine = getEngine();
	if (engine == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::VideoEncoder::encode: encoder session not initialized");
	}
	return encodeRecorded(
		oa::ExecutionSession::forEngine(*engine),
		*this, inRgba, outFrame, inPts);
}
