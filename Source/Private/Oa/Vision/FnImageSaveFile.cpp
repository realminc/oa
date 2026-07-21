// OaFnImage::SaveFile — SaveImage sink (Architecture/OaArchitecture.md §10).
//
// Reads a packed-RGBA8 OaTexture through the engine's synchronous readback
// boundary, then encodes it to disk via stb_image_write. Format is inferred
// from the path extension.
//
// This is the first concrete sink that demonstrates the renderer/sink split
// in code: the texture's producer (loader / renderer / generator) is
// independent of this consumer. A Headless-mode engine can SaveFile without
// any swapchain — proving headless batch / render-farm / CI parity with
// GUI mode.

#include <Oa/Vision/FnImage.h>
#include <Oa/Ui/Image.h>          // OaTexture (was OaUiImage; see Step 2 rename)
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Core/Log.h>

#include "../../ThirdParty/stb/stb_image_write.h"

#include <cstring>
#include <cctype>


namespace {

enum class SaveFmt { Png, Jpg, Bmp, Tga, Unknown };

[[nodiscard]] SaveFmt DetectFormatFromExtension(OaStringView InPath) {
	const char* s = InPath.Data();
	const auto  n = InPath.Size();
	if (n == 0) return SaveFmt::Unknown;

	// Find the last '.', stopping at any path separator.
	size_t dot = n;
	for (size_t i = n; i-- > 0; ) {
		const char c = s[i];
		if (c == '/' or c == '\\') break;
		if (c == '.') { dot = i; break; }
	}
	if (dot == n) return SaveFmt::Unknown;

	auto eqICase = [&](const char* lit) -> bool {
		const size_t litLen = std::strlen(lit);
		if (n - dot != litLen) return false;
		for (size_t i = 0; i < litLen; ++i) {
			const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[dot + i])));
			const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(lit[i])));
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

OaContext& ContextForEngine(OaEngine& InEngine) {
	OaContext* active = OaContext::GetDefaultPtr();
	return active != nullptr and active->GetEngine() == &InEngine
		? *active : InEngine.GetContext();
}

OaStatus ValidateSaveFileRequest(
	OaEngine& InEngine,
	const OaTexture& InTexture,
	OaStringView InPath)
{
	if (not InTexture.IsValid()) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: invalid texture");
	}
	const OaVkBuffer& buffer = InTexture.DeviceBuf;
	if (InTexture.IsImageBacked() or buffer.Buffer == nullptr
		or buffer.Allocation == nullptr or buffer.AliasIdentity != nullptr
		or buffer.IsImported() or buffer.NodeIndex != 0U
		or buffer.AllocatorIdentity != InEngine.Allocator.Allocator) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: texture must be a non-aliased buffer owned by the context engine");
	}
	if (InPath.Empty()) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: empty path");
	}
	if (DetectFormatFromExtension(InPath) == SaveFmt::Unknown) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: unknown extension (expected .png, .jpg, .bmp, .tga)");
	}
	if (InTexture.Width <= 0 or InTexture.Height <= 0) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: texture has zero extent");
	}
	const OaU64 bytes = static_cast<OaU64>(InTexture.Width)
		* static_cast<OaU64>(InTexture.Height) * 4U;
	if (buffer.Size < bytes) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaFnImage::SaveFile: texture buffer smaller than W*H*4");
	}
	if (InPath.Size() >= 1024U) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: path too long");
	}
	return OaStatus::Ok();
}

OaStatus SaveFileReady(
	OaEngine& InEngine,
	const OaTexture&   InTexture,
	OaStringView       InPath
) {
	if (not InTexture.IsValid()) {
		return OaStatus::InvalidArgument("OaFnImage::SaveFile: invalid texture");
	}
	if (InPath.Empty()) {
		return OaStatus::InvalidArgument("OaFnImage::SaveFile: empty path");
	}

	const SaveFmt fmt = DetectFormatFromExtension(InPath);
	if (fmt == SaveFmt::Unknown) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: unknown extension (expected .png, .jpg, .bmp, .tga)");
	}

	const int W = static_cast<int>(InTexture.Width);
	const int H = static_cast<int>(InTexture.Height);
	if (W <= 0 or H <= 0) {
		return OaStatus::InvalidArgument(
			"OaFnImage::SaveFile: texture has zero extent");
	}
	const OaU64 bytes = static_cast<OaU64>(W) * static_cast<OaU64>(H) * 4U;
	if (InTexture.DeviceBuf.Size < bytes) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"OaFnImage::SaveFile: texture buffer smaller than W*H*4");
	}

	// stb_image_write wants a null-terminated C string. Stage into a stack
	// buffer (paths > 1023 chars are pathological and rejected).
	constexpr size_t kPathBufSize = 1024;
	char pathBuf[kPathBufSize];
	if (InPath.Size() >= kPathBufSize) {
		return OaStatus::InvalidArgument("OaFnImage::SaveFile: path too long");
	}
	std::memcpy(pathBuf, InPath.Data(), InPath.Size());
	pathBuf[InPath.Size()] = '\0';

	OaVec<OaU8> pixels;
	pixels.Resize(static_cast<OaI64>(bytes));
	OA_RETURN_IF_ERROR(InEngine.ReadbackBuffer(
		InTexture.DeviceBuf, 0U, pixels.Data(), bytes));

	constexpr int kCompRgba = 4;
	int rc = 0;
	switch (fmt) {
		case SaveFmt::Png:
			rc = stbi_write_png(pathBuf, W, H, kCompRgba, pixels.Data(), W * kCompRgba);
			break;
		case SaveFmt::Jpg:
			rc = stbi_write_jpg(pathBuf, W, H, kCompRgba, pixels.Data(), 90);
			break;
		case SaveFmt::Bmp:
			rc = stbi_write_bmp(pathBuf, W, H, kCompRgba, pixels.Data());
			break;
		case SaveFmt::Tga:
			rc = stbi_write_tga(pathBuf, W, H, kCompRgba, pixels.Data());
			break;
		case SaveFmt::Unknown:
			break;
	}

	if (rc == 0) {
		return OaStatus::Error(OaStatusCode::Internal,
			"OaFnImage::SaveFile: stbi_write_* failed (disk full / permission / bad path?)");
	}

	OA_LOG_INFO(OaLogComponent::App,
		"OaFnImage::SaveFile: %dx%d → %s", W, H, pathBuf);
	return OaStatus::Ok();
}

} // namespace

OaStatus OaFnImage::SaveFile(
	OaContext& InContext,
	const OaTexture& InTexture,
	OaStringView InPath)
{
	OA_RETURN_IF_ERROR(ValidateSaveFileRequest(
		InContext.Engine(), InTexture, InPath));
	OA_RETURN_IF_ERROR(InContext.Execute());
	OA_RETURN_IF_ERROR(InContext.Sync());
	return SaveFileReady(InContext.Engine(), InTexture, InPath);
}

OaStatus OaFnImage::SaveFile(
	OaEngine& InEngine,
	const OaTexture& InTexture,
	OaStringView InPath)
{
	return SaveFile(ContextForEngine(InEngine), InTexture, InPath);
}
