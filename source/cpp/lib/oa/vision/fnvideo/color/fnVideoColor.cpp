// Stateless oa::FnVideo frame views.

#include <oa/vision/fnVideo.h>
#include <oa/runtime/texture.h>

#include "../../../runtime/textureAccess.h"

namespace oa {

namespace FnVideo {

oa::Result<VideoFrame> fromTexture(
	const oa::Texture& inTexture,
	oa::U64 inPts,
	oa::Event inReady)
{
	if (not inTexture.isValid() or inTexture.width() <= 0 or inTexture.height() <= 0) {
		return oa::Status::error(oa::StatusCode::InvalidArgument,
			"oa::FnVideo::fromTexture requires a valid texture with positive extent");
	}
	VideoFrame frame = {};
	frame.width = static_cast<oa::U32>(inTexture.width());
	frame.height = static_cast<oa::U32>(inTexture.height());
	frame.presentationTimestamp = inPts;
	frame.isRgb = true;
	frame.colorSpace = oa::YCbCrModel::BT709;
	frame.fullRange = true;
	frame.ready = inReady;
	if (inTexture.isImageBacked()) {
		if (oa::TextureAccess::view(inTexture) == VK_NULL_HANDLE
			or oa::TextureAccess::format(inTexture) == VK_FORMAT_UNDEFINED
			or oa::TextureAccess::layout(inTexture) == VK_IMAGE_LAYOUT_UNDEFINED) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"oa::FnVideo::fromTexture image targets require view, format and current layout");
		}
		const VkFormat format = oa::TextureAccess::format(inTexture);
		if (format != VK_FORMAT_R8G8B8A8_UNORM
			and format != VK_FORMAT_B8G8R8A8_UNORM) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"oa::FnVideo::fromTexture supports RGBA8/BGRA8 UNORM render targets");
		}
		frame.resource = VideoFrameResource::Image;
		frame.image = oa::TextureAccess::image(inTexture);
		frame.imageView = oa::TextureAccess::view(inTexture);
		frame.format = format;
		frame.layout = oa::TextureAccess::layout(inTexture);
	} else {
		const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTexture);
		if (buffer == nullptr or buffer->buffer == nullptr) {
			return oa::Status::error(oa::StatusCode::InvalidArgument,
				"oa::FnVideo::fromTexture buffer target has no device buffer");
		}
		frame.resource = VideoFrameResource::Buffer;
		frame.buffer = buffer;
		frame.format = VK_FORMAT_R8G8B8A8_UNORM;
	}
	return frame;
}

} // namespace FnVideo

} // namespace oa
