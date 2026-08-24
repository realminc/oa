#include <oa/ui/style.h>

#include <array>
#include <cmath>

namespace {

[[nodiscard]] bool isUnitColor(const oa::Color& inColor) noexcept {
	return std::isfinite(inColor.r) && std::isfinite(inColor.g)
		&& std::isfinite(inColor.b) && std::isfinite(inColor.a)
		&& inColor.r >= 0.0F && inColor.r <= 1.0F
		&& inColor.g >= 0.0F && inColor.g <= 1.0F
		&& inColor.b >= 0.0F && inColor.b <= 1.0F
		&& inColor.a >= 0.0F && inColor.a <= 1.0F;
}

[[nodiscard]] constexpr oa::Color hex(oa::U32 inRgba) noexcept {
	return oa::Color::fromU32(inRgba);
}

} // namespace

oa::Status oa::UiStyle::validate() const {
	const std::array<oa::F32, 9> geometry{{
		cornerRadius,
		borderWidth,
		shadowBlur,
		shadowOffset,
		fontSize,
		itemSpacing,
		padding,
		framePaddingX,
		framePaddingY,
	}};
	for (const oa::F32 value : geometry) {
		if (!std::isfinite(value)) {
			return oa::Status::invalidArgument(
				"oa::UiStyle geometry must be finite");
		}
	}
	if (cornerRadius < 0.0F || borderWidth < 0.0F || shadowBlur < 0.0F
		|| shadowOffset < 0.0F || fontSize <= 0.0F || itemSpacing < 0.0F
		|| padding < 0.0F || framePaddingX < 0.0F || framePaddingY < 0.0F)
	{
		return oa::Status::invalidArgument(
			"oa::UiStyle requires positive font size and non-negative geometry");
	}

	const std::array<oa::Color, 17> colors{{
		background,
		surface,
		surfaceHover,
		surfaceActive,
		borderSubtle,
		border,
		borderStrong,
		text,
		textSecondary,
		textMuted,
		textDisabled,
		accent,
		accentHover,
		accentActive,
		success,
		warning,
		error,
	}};
	for (const oa::Color& color : colors) {
		if (!isUnitColor(color)) {
			return oa::Status::invalidArgument(
				"oa::UiStyle colors must be finite RGBA values in [0,1]");
		}
	}
	return oa::Status::ok();
}

oa::UiStyle oa::UiStyle::realmDark() {
	return {};
}

oa::UiStyle oa::UiStyle::realmLight() {
	oa::UiStyle style;
	style.background = hex(0xFAFAFAFFU);
	style.surface = hex(0xFFFFFFFFU);
	style.surfaceHover = hex(0xF5F5F5FFU);
	style.surfaceActive = hex(0xE5E7EBFFU);
	style.borderSubtle = hex(0x1118270FU);
	style.border = hex(0x1118271AU);
	style.borderStrong = hex(0x11182726U);
	style.text = hex(0x171717FFU);
	style.textSecondary = hex(0x404040FFU);
	style.textMuted = hex(0x737373FFU);
	style.textDisabled = hex(0xA3A3A3FFU);
	style.accent = hex(0x4F46E5FFU);
	style.accentHover = hex(0x4338CAFFU);
	style.accentActive = hex(0x3730A3FFU);
	style.success = hex(0x15803DFFU);
	style.warning = hex(0xB45309FFU);
	style.error = hex(0xDC2626FFU);
	return style;
}

oa::UiStyle oa::UiStyle::editorDark() {
	oa::UiStyle style;
	style.cornerRadius = 3.0F;
	style.shadowBlur = 6.0F;
	style.fontSize = 15.0F;
	style.itemSpacing = 4.0F;
	style.padding = 8.0F;
	style.framePaddingX = 6.0F;
	style.framePaddingY = 4.0F;
	style.background = hex(0x181818FFU);
	style.surface = hex(0x313131FFU);
	style.surfaceHover = hex(0x3C3C3CFFU);
	style.surfaceActive = hex(0x2B2B2BFFU);
	style.borderSubtle = hex(0x2B2B2B80U);
	style.border = hex(0x2B2B2B99U);
	style.borderStrong = hex(0x3C3C3CFFU);
	style.text = hex(0xCCCCCCFFU);
	style.textSecondary = hex(0x9D9D9DFFU);
	style.textMuted = hex(0x868686FFU);
	style.textDisabled = hex(0x6E7681FFU);
	style.accent = hex(0x0078D4FFU);
	style.accentHover = hex(0x4DAAFCFFU);
	style.accentActive = hex(0x026EC1FFU);
	style.success = hex(0x3FB950FFU);
	style.warning = hex(0xCCA700FFU);
	style.error = hex(0xF14C4CFFU);
	return style;
}

oa::UiStyle oa::UiStyle::editorLight() {
	oa::UiStyle style = editorDark();
	style.background = hex(0xF8F8F8FFU);
	style.surface = hex(0xE5E5E5FFU);
	style.surfaceHover = hex(0xCCCCCCFFU);
	style.surfaceActive = hex(0xE8E8E8FFU);
	style.borderSubtle = hex(0xE5E5E580U);
	style.border = hex(0xCECECEFFU);
	style.borderStrong = hex(0xA8A8A8FFU);
	style.text = hex(0x3B3B3BFFU);
	style.textSecondary = hex(0x616161FFU);
	style.textMuted = hex(0x868686FFU);
	style.textDisabled = hex(0xA0A0A0FFU);
	style.accent = hex(0x005FB8FFU);
	style.accentHover = hex(0x0078D4FFU);
	style.accentActive = hex(0x0258A8FFU);
	style.success = hex(0x16825DFFU);
	style.warning = hex(0x9D5D00FFU);
	style.error = hex(0xC72E0FFFU);
	return style;
}
