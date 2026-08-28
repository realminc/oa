// oa::FnImage::SaveTextureFile — texture file sink.
//
// Reads a packed-RGBA8 oa::Texture through the engine's synchronous readback
// boundary, then encodes it to disk via stb_image_write. format is inferred
// from the path extension.
//
// This is the first concrete sink that demonstrates the renderer/sink split
// in code: the texture's producer (loader / renderer / generator) is
// independent of this consumer. A Headless-mode engine can SaveFile without
// any swapchain — proving headless batch / render-farm / CI parity with
// GUI mode.

#include <oa/vision/fnImage.h>
#include <oa/runtime/texture.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/allocator.h>
#include <oa/core/log.h>
#include <oa/core/std/memory.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/limits.h>

#include "../runtime/textureAccess.h"

#include "../../thirdparty/stb/stb_image_write.h"

namespace {

enum class SaveFmt { Png, Jpg, Bmp, Tga, Unknown };

[[nodiscard]] SaveFmt detectFormatFromExtension(oa::StringView inPath) {
	const char* s = inPath.data();
	const auto  n = inPath.size();
	if (n == 0) return SaveFmt::Unknown;

	// find the last '.', stopping at any path separator.
	oa::Usize dot = n;
	for (oa::Usize i = n; i-- > 0; ) {
		const char c = s[i];
		if (c == '/' or c == '\\') break;
		if (c == '.') { dot = i; break; }
	}
	if (dot == n) return SaveFmt::Unknown;

	auto eqICase = [&](const char* lit) -> bool {
		auto lowerAscii = [](char inValue) {
			return inValue >= 'A' and inValue <= 'Z'
				? static_cast<char>(inValue + ('a' - 'A')) : inValue;
		};
		const oa::Usize litLen = oa::strlen(lit);
		if (n - dot != litLen) return false;
		for (oa::Usize i = 0; i < litLen; ++i) {
			const char a = lowerAscii(s[dot + i]);
			const char b = lowerAscii(lit[i]);
			if (a != b) return false;
		}
		return true;
	};

	if (eqICase(".png"))                       return SaveFmt::Png;
	if (eqICase(".jpg") or eqICase(".jpeg"))   return SaveFmt::Jpg;
	if (eqICase(".bmp"))                       return SaveFmt::Bmp;
	if (eqICase(".tga"))                       return SaveFmt::Tga;
	return SaveFmt::Unknown;
}

} // namespace


namespace {

oa::Status validateSaveFileRequest(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	oa::StringView inPath)
{
	if (not inTexture.isValid()) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: invalid texture");
	}
	const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTexture);
	if (inTexture.isImageBacked() or buffer == nullptr or buffer->buffer == nullptr
		or buffer->allocation == nullptr or buffer->aliasIdentity != nullptr
		or oa::TextureAccess::engine(inTexture) != &inEngine
		or buffer->allocatorIdentity != oa::EngineAllocatorAccess::get(inEngine).allocator) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: texture must be a non-aliased buffer owned by the context engine");
	}
	if (inPath.empty()) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: empty path");
	}
	if (detectFormatFromExtension(inPath) == SaveFmt::Unknown) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: unknown extension (expected .png, .jpg, .bmp, .tga)");
	}
	if (inTexture.width() <= 0 or inTexture.height() <= 0) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: texture has zero extent");
	}
	const oa::U64 bytes = static_cast<oa::U64>(inTexture.width())
		* static_cast<oa::U64>(inTexture.height()) * 4U;
	if (buffer->size < bytes) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::FnImage::saveTextureFile: texture buffer smaller than W*H*4");
	}
	if (inPath.size() >= 1024U) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveTextureFile: path too long");
	}
	return oa::Status::ok();
}

oa::Status saveRgbaFileReady(
	oa::Span<const oa::U8> inRgba,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::StringView inPath,
	oa::U32 inQuality) {
	if (inPath.empty()) {
		return oa::Status::invalidArgument("oa::FnImage::saveRgbaFile: empty path");
	}

	const SaveFmt fmt = detectFormatFromExtension(inPath);
	if (fmt == SaveFmt::Unknown) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveRgbaFile: unknown extension (expected .png, .jpg, .bmp, .tga)");
	}
	if (inWidth == 0U or inHeight == 0U
		or inWidth > static_cast<oa::U32>(oa::Limits<int>::max())
		or inHeight > static_cast<oa::U32>(oa::Limits<int>::max())) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveRgbaFile: extent must be positive and int-addressable");
	}
	const oa::U64 pixels = static_cast<oa::U64>(inWidth) * inHeight;
	if (pixels > oa::Limits<oa::U64>::max() / 4U
		or pixels * 4U != inRgba.size()) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveRgbaFile: byte span must equal width*height*4");
	}
	if (inQuality == 0U or inQuality > 100U) {
		return oa::Status::invalidArgument(
			"oa::FnImage::saveRgbaFile: quality must be in [1, 100]");
	}

	// stb_image_write wants a null-terminated C string. stage into a stack
	// buffer (paths > 1023 chars are pathological and rejected).
	constexpr oa::Usize kPathBufSize = 1024;
	char pathBuf[kPathBufSize];
	if (inPath.size() >= kPathBufSize) {
		return oa::Status::invalidArgument("oa::FnImage::saveRgbaFile: path too long");
	}
	oa::memcpy(pathBuf, inPath.data(), inPath.size());
	pathBuf[inPath.size()] = '\0';

	const int W = static_cast<int>(inWidth);
	const int H = static_cast<int>(inHeight);
	constexpr int kCompRgba = 4;
	int rc = 0;
	switch (fmt) {
		case SaveFmt::Png:
			rc = stbi_write_png(pathBuf, W, H, kCompRgba, inRgba.data(), W * kCompRgba);
			break;
		case SaveFmt::Jpg:
			rc = stbi_write_jpg(
				pathBuf, W, H, kCompRgba, inRgba.data(),
				static_cast<int>(inQuality));
			break;
		case SaveFmt::Bmp:
			rc = stbi_write_bmp(pathBuf, W, H, kCompRgba, inRgba.data());
			break;
		case SaveFmt::Tga:
			rc = stbi_write_tga(pathBuf, W, H, kCompRgba, inRgba.data());
			break;
		case SaveFmt::Unknown:
			break;
	}

	if (rc == 0) {
		return oa::Status::error(oa::StatusCode::Internal,
			"oa::FnImage::saveRgbaFile: stbi_write_* failed (disk full / permission / bad path?)");
	}

	OaLogInfo(oa::LogComponent::Vision,
		"oa::FnImage::saveRgbaFile: %dx%d → %s", W, H, pathBuf);
	return oa::Status::ok();
}

oa::Status saveTextureFileReady(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	oa::StringView inPath) {
	const oa::U64 bytes = static_cast<oa::U64>(inTexture.width())
		* static_cast<oa::U64>(inTexture.height()) * 4U;
	const oavk::Buffer* buffer = oa::TextureAccess::buffer(inTexture);
	if (buffer == nullptr or buffer->size < bytes) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::FnImage::saveTextureFile: texture buffer smaller than W*H*4");
	}
	oa::Vector<oa::U8> pixels;
	pixels.resize(static_cast<oa::Usize>(bytes));
	OA_RETURN_IF_ERROR(oa::EngineResourceAccess::readbackBuffer(
		inEngine, *buffer, 0U, pixels.data(), bytes));
	return saveRgbaFileReady(
		oa::Span<const oa::U8>(pixels.data(), pixels.size()),
		static_cast<oa::U32>(inTexture.width()),
		static_cast<oa::U32>(inTexture.height()),
		inPath,
		90U);
}

} // namespace

oa::Status oa::FnImage::saveTextureFile(
	oa::Engine& inEngine,
	const oa::Texture& inTexture,
	oa::StringView inPath)
{
	auto& context = oa::ExecutionSession::forEngine(inEngine);
	OA_RETURN_IF_ERROR(validateSaveFileRequest(
		inEngine, inTexture, inPath));
	OA_RETURN_IF_ERROR(context.submitAndWait());
	return saveTextureFileReady(inEngine, inTexture, inPath);
}

oa::Status oa::FnImage::saveRgbaFile(
	oa::Span<const oa::U8> inRgba,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::StringView inPath,
	oa::U32 inQuality) {
	return saveRgbaFileReady(
		inRgba, inWidth, inHeight, inPath, inQuality);
}
