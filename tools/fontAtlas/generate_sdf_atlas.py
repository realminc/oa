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
    bearing_x: float
    bearing_y: float
    advance: float
    atlas_x: int = 0
    atlas_y: int = 0


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _i16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">h", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def unicode_cmap(path: Path) -> set[int]:
    """Return Unicode scalars with a non-zero nominal glyph in a TrueType cmap."""
    data = path.read_bytes()
    table_count = _u16(data, 4)
    cmap_offset = None
    for index in range(table_count):
        record = 12 + index * 16
        if data[record:record + 4] == b"cmap":
            cmap_offset = _u32(data, record + 8)
            break
    if cmap_offset is None:
        raise ValueError(f"{path}: missing cmap table")

    result: set[int] = set()
    subtable_count = _u16(data, cmap_offset + 2)
    seen: set[int] = set()
    for index in range(subtable_count):
        record = cmap_offset + 4 + index * 8
        platform = _u16(data, record)
        encoding = _u16(data, record + 2)
        if platform != 0 and not (platform == 3 and encoding in (1, 10)):
            continue
        subtable = cmap_offset + _u32(data, record + 4)
        if subtable in seen:
            continue
        seen.add(subtable)
        format_id = _u16(data, subtable)
        if format_id == 4:
            segment_count = _u16(data, subtable + 6) // 2
            end_codes = subtable + 14
            start_codes = end_codes + segment_count * 2 + 2
            deltas = start_codes + segment_count * 2
            range_offsets = deltas + segment_count * 2
            for segment in range(segment_count):
                start = _u16(data, start_codes + segment * 2)
                end = _u16(data, end_codes + segment * 2)
                delta = _i16(data, deltas + segment * 2)
                range_offset = _u16(data, range_offsets + segment * 2)
                for codepoint in range(start, min(end, 0xFFFE) + 1):
                    if range_offset == 0:
                        glyph_id = (codepoint + delta) & 0xFFFF
                    else:
                        glyph_address = (
                            range_offsets + segment * 2 + range_offset
                            + (codepoint - start) * 2
                        )
                        glyph_id = _u16(data, glyph_address)
                        if glyph_id:
                            glyph_id = (glyph_id + delta) & 0xFFFF
                    if glyph_id:
                        result.add(codepoint)
        elif format_id == 12:
            group_count = _u32(data, subtable + 12)
            for group in range(group_count):
                group_offset = subtable + 16 + group * 12
                start = _u32(data, group_offset)
                end = min(_u32(data, group_offset + 4), 0x10FFFF)
                start_glyph = _u32(data, group_offset + 8)
                for codepoint in range(start, end + 1):
                    if start_glyph + codepoint - start:
                        result.add(codepoint)
    return result


def selected_codepoints(*font_codepoints: set[int]) -> tuple[int, ...]:
    candidates = {
        codepoint
        for first, last in CANDIDATE_RANGES
        for codepoint in range(first, last + 1)
    }
    supported = set().union(*font_codepoints)
    return tuple(sorted(candidates & supported))


def rasterize(
    font_id: int,
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
                glyphs.append(Glyph(font_id, strike, codepoint,
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
                    font=font_id,
                    strike=strike,
                    codepoint=codepoint,
                    mask=mask,
                    bearing_x=float(left),
                    bearing_y=float(-top),
                    advance=float(face.getlength(character)),
                )
            )
    return glyphs


def pack(glyphs: list[Glyph], width: int, padding: int) -> Image.Image:
    cursor_x = padding
    cursor_y = padding
    row_height = 0
    for glyph in glyphs:
        packed_width = glyph.mask.width + padding * 2
        packed_height = glyph.mask.height + padding * 2
        if cursor_x + packed_width > width:
            cursor_x = padding
            cursor_y += row_height
            row_height = 0
        glyph.atlas_x = cursor_x + padding
        glyph.atlas_y = cursor_y + padding
        cursor_x += packed_width
        row_height = max(row_height, packed_height)

    height = cursor_y + row_height + padding
    atlas = Image.new("L", (width, height), 0)
    for glyph in glyphs:
        atlas.paste(glyph.mask, (glyph.atlas_x, glyph.atlas_y))
    return atlas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cpp(
    output: Path,
    atlas: Image.Image,
    glyphs: list[Glyph],
    strikes: tuple[int, ...],
    sans_font: Path,
    mono_font: Path,
    codepoints: tuple[int, ...],
    font_codepoints: tuple[set[int], set[int]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = list(atlas.getdata())
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("// Generated by tools/fontAtlas/generate_sdf_atlas.py. DO NOT EDIT.\n")
        stream.write(f"// Pillow version: {PILLOW_VERSION}\n")
        stream.write(f"// FreeType version: {features.version_module('freetype2')}\n")
        stream.write(f'// IBM Plex Sans SHA-256: {sha256(sans_font)}\n')
        stream.write(f'// Intel One Mono SHA-256: {sha256(mono_font)}\n')
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
        for supported in font_codepoints:
            stream.write("\t{")
            stream.write(", ".join(
                "true" if codepoint in supported else "false"
                for codepoint in codepoints
            ))
            stream.write("},\n")
        stream.write("};\n")
        for symbol, font_path in (
            ("kTextSansFontBytes", sans_font),
            ("kTextMonoFontBytes", mono_font),
        ):
            font_bytes = font_path.read_bytes()
            stream.write(f"static constexpr oa::U8 {symbol}[] = {{\n")
            for index in range(0, len(font_bytes), 24):
                stream.write("\t" + ", ".join(
                    str(value) for value in font_bytes[index:index + 24]
                ) + ",\n")
            stream.write("};\n")
        stream.write("static constexpr OaGeneratedGlyph kTextGlyphs[] = {\n")
        for glyph in glyphs:
            stream.write(
                "\t{%d, %d, %d, %d, %d, %d, %d, %.3fF, %.3fF, %.3fF},\n"
                % (
                    glyph.font,
                    glyph.strike,
                    glyph.codepoint,
                    glyph.atlas_x,
                    glyph.atlas_y,
                    glyph.mask.width,
                    glyph.mask.height,
                    glyph.bearing_x,
                    glyph.bearing_y,
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
    parser.add_argument("--sans-font", required=True, type=Path)
    parser.add_argument("--mono-font", required=True, type=Path)
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

    sans_codepoints = unicode_cmap(args.sans_font)
    mono_codepoints = unicode_cmap(args.mono_font)
    codepoints = selected_codepoints(sans_codepoints, mono_codepoints)
    glyphs = rasterize(
        0, args.sans_font, strikes, codepoints, sans_codepoints)
    glyphs.extend(rasterize(
        1, args.mono_font, strikes, codepoints, mono_codepoints))
    atlas = pack(glyphs, args.width, args.padding)
    write_cpp(
        args.output, atlas, glyphs, strikes, args.sans_font, args.mono_font,
        codepoints, (sans_codepoints, mono_codepoints))


if __name__ == "__main__":
    main()
