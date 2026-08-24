#!/usr/bin/env python3
"""Generate OA's deterministic hinted grayscale UI-text atlas.

The generated atlas contains exact pixel-size strikes. Runtime code therefore
does not scale one large SDF down for every UI label, and does not need to load
font files or initialize a second graphics/runtime subsystem.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import struct

from PIL import Image, ImageDraw, ImageFont, features, __version__ as PILLOW_VERSION


# Coverage is deliberately bounded to scripts and technical symbols present in
# at least one of OA's two vendored faces. The runtime performs explicit
# per-codepoint fallback; it never treats a font's .notdef box as coverage.
CANDIDATE_RANGES = (
    (0x0020, 0x007E),  # Basic Latin
    (0x00A0, 0x024F),  # Latin-1 + Latin Extended A/B
    (0x02B0, 0x036F),  # spacing modifiers + combining marks
    (0x0370, 0x052F),  # Greek, Coptic, Cyrillic, Cyrillic Supplement
    (0x1E00, 0x1EFF),  # Latin Extended Additional
    (0x2000, 0x206F),  # General Punctuation
    (0x2070, 0x209F),  # Superscripts and Subscripts
    (0x20A0, 0x20CF),  # Currency Symbols
    (0x2100, 0x214F),  # Letterlike Symbols
    (0x2150, 0x218F),  # Number Forms
    (0x2190, 0x22FF),  # Arrows + Mathematical Operators
    (0x2300, 0x23FF),  # Miscellaneous Technical
    (0x2500, 0x259F),  # Box Drawing + Block Elements
    (0x25A0, 0x26FF),  # Geometric Shapes + Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0xFB00, 0xFB06),  # Latin presentation ligatures emitted by shaping
    (0xFFFD, 0xFFFD),  # visible replacement character
)
DEFAULT_STRIKES = (9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32)


@dataclass
class Glyph:
    font: int
    strike: int
    codepoint: int
    mask: Image.Image
    bearingX: float
    bearingY: float
    advance: float
    atlasX: int = 0
    atlasY: int = 0


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _i16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">h", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def unicodeCmap(path: Path) -> set[int]:
    """Return Unicode scalars with a non-zero nominal glyph in a TrueType cmap."""
    data = path.read_bytes()
    tableCount = _u16(data, 4)
    cmapOffset = None
    for index in range(tableCount):
        record = 12 + index * 16
        if data[record:record + 4] == b"cmap":
            cmapOffset = _u32(data, record + 8)
            break
    if cmapOffset is None:
        raise ValueError(f"{path}: missing cmap table")

    result: set[int] = set()
    subtableCount = _u16(data, cmapOffset + 2)
    seen: set[int] = set()
    for index in range(subtableCount):
        record = cmapOffset + 4 + index * 8
        platform = _u16(data, record)
        encoding = _u16(data, record + 2)
        if platform != 0 and not (platform == 3 and encoding in (1, 10)):
            continue
        subtable = cmapOffset + _u32(data, record + 4)
        if subtable in seen:
            continue
        seen.add(subtable)
        formatId = _u16(data, subtable)
        if formatId == 4:
            segmentCount = _u16(data, subtable + 6) // 2
            endCodes = subtable + 14
            startCodes = endCodes + segmentCount * 2 + 2
            deltas = startCodes + segmentCount * 2
            rangeOffsets = deltas + segmentCount * 2
            for segment in range(segmentCount):
                start = _u16(data, startCodes + segment * 2)
                end = _u16(data, endCodes + segment * 2)
                delta = _i16(data, deltas + segment * 2)
                rangeOffset = _u16(data, rangeOffsets + segment * 2)
                for codepoint in range(start, min(end, 0xFFFE) + 1):
                    if rangeOffset == 0:
                        glyphId = (codepoint + delta) & 0xFFFF
                    else:
                        glyphAddress = (
                            rangeOffsets + segment * 2 + rangeOffset
                            + (codepoint - start) * 2
                        )
                        glyphId = _u16(data, glyphAddress)
                        if glyphId:
                            glyphId = (glyphId + delta) & 0xFFFF
                    if glyphId:
                        result.add(codepoint)
        elif formatId == 12:
            groupCount = _u32(data, subtable + 12)
            for group in range(groupCount):
                groupOffset = subtable + 16 + group * 12
                start = _u32(data, groupOffset)
                end = min(_u32(data, groupOffset + 4), 0x10FFFF)
                startGlyph = _u32(data, groupOffset + 8)
                for codepoint in range(start, end + 1):
                    if startGlyph + codepoint - start:
                        result.add(codepoint)
    return result


def selectedCodepoints(*fontCodepoints: set[int]) -> tuple[int, ...]:
    candidates = {
        codepoint
        for first, last in CANDIDATE_RANGES
        for codepoint in range(first, last + 1)
    }
    supported = set().union(*fontCodepoints)
    return tuple(sorted(candidates & supported))


def rasterize(
    fontId: int,
    path: Path,
    strikes: tuple[int, ...],
    codepoints: tuple[int, ...],
    supported: set[int],
) -> list[Glyph]:
    glyphs: list[Glyph] = []
    for strike in strikes:
        face = ImageFont.truetype(str(path), strike)
        for codepoint in codepoints:
            if codepoint not in supported:
                glyphs.append(Glyph(fontId, strike, codepoint,
                                    Image.new("L", (1, 1), 0), 0.0, 0.0, 0.0))
                continue
            character = chr(codepoint)
            left, top, right, bottom = face.getbbox(character, anchor="ls")
            width = max(1, right - left)
            height = max(1, bottom - top)
            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).text(
                (-left, -top), character, font=face, fill=255, anchor="ls"
            )
            glyphs.append(
                Glyph(
                    font=fontId,
                    strike=strike,
                    codepoint=codepoint,
                    mask=mask,
                    bearingX=float(left),
                    bearingY=float(-top),
                    advance=float(face.getlength(character)),
                )
            )
    return glyphs


def pack(glyphs: list[Glyph], width: int, padding: int) -> Image.Image:
    cursorX = padding
    cursorY = padding
    rowHeight = 0
    for glyph in glyphs:
        packedWidth = glyph.mask.width + padding * 2
        packedHeight = glyph.mask.height + padding * 2
        if cursorX + packedWidth > width:
            cursorX = padding
            cursorY += rowHeight
            rowHeight = 0
        glyph.atlasX = cursorX + padding
        glyph.atlasY = cursorY + padding
        cursorX += packedWidth
        rowHeight = max(rowHeight, packedHeight)

    height = cursorY + rowHeight + padding
    atlas = Image.new("L", (width, height), 0)
    for glyph in glyphs:
        atlas.paste(glyph.mask, (glyph.atlasX, glyph.atlasY))
    return atlas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def writeCpp(
    output: Path,
    atlas: Image.Image,
    glyphs: list[Glyph],
    strikes: tuple[int, ...],
    sansFont: Path,
    monoFont: Path,
    codepoints: tuple[int, ...],
    fontCodepoints: tuple[set[int], set[int]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = list(atlas.getdata())
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("// Generated by tools/gen/font/generateAtlas.py. DO NOT EDIT.\n")
        stream.write(f"// Pillow version: {PILLOW_VERSION}\n")
        stream.write(f"// FreeType version: {features.version_module('freetype2')}\n")
        stream.write(f'// IBM Plex Sans SHA-256: {sha256(sansFont)}\n')
        stream.write(f'// Intel One Mono SHA-256: {sha256(monoFont)}\n')
        stream.write(f"static constexpr oa::U32 kTextAtlasWidth = {atlas.width};\n")
        stream.write(f"static constexpr oa::U32 kTextAtlasHeight = {atlas.height};\n")
        stream.write(f"static constexpr oa::U32 kTextFontCount = 2;\n")
        stream.write(f"static constexpr oa::U32 kTextStrikeCount = {len(strikes)};\n")
        stream.write(f"static constexpr oa::U32 kTextCodepointCount = {len(codepoints)};\n")
        stream.write(f"static constexpr oa::U32 kTextGlyphCount = {len(glyphs)};\n")
        stream.write("static constexpr oa::U32 kTextStrikeSizes[] = {")
        stream.write(", ".join(str(value) for value in strikes))
        stream.write("};\n")
        stream.write("static constexpr oa::U32 kTextCodepoints[] = {")
        stream.write(", ".join(str(value) for value in codepoints))
        stream.write("};\n")
        stream.write("static constexpr bool kTextCodepointSupport[][kTextCodepointCount] = {\n")
        for supported in fontCodepoints:
            stream.write("\t{")
            stream.write(", ".join(
                "true" if codepoint in supported else "false"
                for codepoint in codepoints
            ))
            stream.write("},\n")
        stream.write("};\n")
        for symbol, fontPath in (
            ("kTextSansFontBytes", sansFont),
            ("kTextMonoFontBytes", monoFont),
        ):
            fontBytes = fontPath.read_bytes()
            stream.write(f"static constexpr oa::U8 {symbol}[] = {{\n")
            for index in range(0, len(fontBytes), 24):
                stream.write("\t" + ", ".join(
                    str(value) for value in fontBytes[index:index + 24]
                ) + ",\n")
            stream.write("};\n")
        stream.write("static constexpr GeneratedGlyph kTextGlyphs[] = {\n")
        for glyph in glyphs:
            stream.write(
                "\t{%d, %d, %d, %d, %d, %d, %d, %.3fF, %.3fF, %.3fF},\n"
                % (
                    glyph.font,
                    glyph.strike,
                    glyph.codepoint,
                    glyph.atlasX,
                    glyph.atlasY,
                    glyph.mask.width,
                    glyph.mask.height,
                    glyph.bearingX,
                    glyph.bearingY,
                    glyph.advance,
                )
            )
        stream.write("};\n")
        stream.write("static constexpr oa::U8 kTextAtlasPixels[] = {\n")
        for index in range(0, len(pixels), 24):
            stream.write("\t" + ", ".join(str(v) for v in pixels[index:index + 24]) + ",\n")
        stream.write("};\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sans-font", dest="sansFont", required=True, type=Path)
    parser.add_argument("--mono-font", dest="monoFont", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strikes", default=",".join(map(str, DEFAULT_STRIKES)))
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--padding", type=int, default=1)
    args = parser.parse_args()

    strikes = tuple(sorted({int(value) for value in args.strikes.split(",")}))
    if not strikes or strikes[0] <= 0:
        parser.error("--strikes must contain positive pixel sizes")
    if args.width <= 0 or args.padding < 1:
        parser.error("--width must be positive and --padding must be at least one")

    sansCodepoints = unicodeCmap(args.sansFont)
    monoCodepoints = unicodeCmap(args.monoFont)
    codepoints = selectedCodepoints(sansCodepoints, monoCodepoints)
    glyphs = rasterize(
        0, args.sansFont, strikes, codepoints, sansCodepoints)
    glyphs.extend(rasterize(
        1, args.monoFont, strikes, codepoints, monoCodepoints))
    atlas = pack(glyphs, args.width, args.padding)
    writeCpp(
        args.output, atlas, glyphs, strikes, args.sansFont, args.monoFont,
        codepoints, (sansCodepoints, monoCodepoints))


if __name__ == "__main__":
    main()
