#!/usr/bin/env python3
"""Generate OA's deterministic compact hinted text-atlas assets.

The raster and selection algorithm is the OA C++ donor algorithm.  Rust stores
the result as compact binary payloads instead of a C++ integer initializer.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import struct

from PIL import Image, ImageDraw, ImageFont, features, __version__ as PILLOW_VERSION

CANDIDATE_RANGES = (
    (0x0020, 0x007E), (0x00A0, 0x024F), (0x02B0, 0x036F),
    (0x0370, 0x052F), (0x1E00, 0x1EFF), (0x2000, 0x206F),
    (0x2070, 0x209F), (0x20A0, 0x20CF), (0x2100, 0x214F),
    (0x2150, 0x218F), (0x2190, 0x22FF), (0x2300, 0x23FF),
    (0x2500, 0x259F), (0x25A0, 0x26FF), (0x2700, 0x27BF),
    (0xFB00, 0xFB06), (0xFFFD, 0xFFFD),
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


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def i16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">h", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def unicode_cmap(path: Path) -> set[int]:
    data = path.read_bytes()
    table_count = u16(data, 4)
    cmap_offset = next((u32(data, 12 + i * 16 + 8) for i in range(table_count)
                        if data[12 + i * 16:16 + i * 16] == b"cmap"), None)
    if cmap_offset is None:
        raise ValueError(f"{path}: missing cmap table")
    result: set[int] = set()
    seen: set[int] = set()
    for index in range(u16(data, cmap_offset + 2)):
        record = cmap_offset + 4 + index * 8
        platform, encoding = u16(data, record), u16(data, record + 2)
        if platform != 0 and not (platform == 3 and encoding in (1, 10)):
            continue
        subtable = cmap_offset + u32(data, record + 4)
        if subtable in seen:
            continue
        seen.add(subtable)
        format_id = u16(data, subtable)
        if format_id == 4:
            count = u16(data, subtable + 6) // 2
            ends, starts = subtable + 14, subtable + 14 + count * 2 + 2
            deltas, ranges = starts + count * 2, starts + count * 4
            for segment in range(count):
                start, end = u16(data, starts + segment * 2), u16(data, ends + segment * 2)
                delta, range_offset = i16(data, deltas + segment * 2), u16(data, ranges + segment * 2)
                for cp in range(start, min(end, 0xFFFE) + 1):
                    if range_offset == 0:
                        glyph_id = (cp + delta) & 0xFFFF
                    else:
                        glyph_id = u16(data, ranges + segment * 2 + range_offset + (cp - start) * 2)
                        glyph_id = (glyph_id + delta) & 0xFFFF if glyph_id else 0
                    if glyph_id:
                        result.add(cp)
        elif format_id == 12:
            for group in range(u32(data, subtable + 12)):
                offset = subtable + 16 + group * 12
                start, end, first = u32(data, offset), min(u32(data, offset + 4), 0x10FFFF), u32(data, offset + 8)
                result.update(cp for cp in range(start, end + 1) if first + cp - start)
    return result


def rasterize(font_id: int, path: Path, strikes: tuple[int, ...], codepoints: tuple[int, ...], supported: set[int]) -> list[Glyph]:
    glyphs = []
    for strike in strikes:
        face = ImageFont.truetype(str(path), strike)
        for cp in codepoints:
            if cp not in supported:
                glyphs.append(Glyph(font_id, strike, cp, Image.new("L", (1, 1), 0), 0.0, 0.0, 0.0))
                continue
            char = chr(cp)
            left, top, right, bottom = face.getbbox(char, anchor="ls")
            mask = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
            ImageDraw.Draw(mask).text((-left, -top), char, font=face, fill=255, anchor="ls")
            glyphs.append(Glyph(font_id, strike, cp, mask, float(left), float(-top), float(face.getlength(char))))
    return glyphs


def pack(glyphs: list[Glyph], width: int, padding: int) -> Image.Image:
    x = y = padding
    row_height = 0
    for glyph in glyphs:
        packed_width, packed_height = glyph.mask.width + 2 * padding, glyph.mask.height + 2 * padding
        if x + packed_width > width:
            x, y, row_height = padding, y + row_height, 0
        glyph.atlas_x, glyph.atlas_y = x + padding, y + padding
        x += packed_width
        row_height = max(row_height, packed_height)
    atlas = Image.new("L", (width, y + row_height + padding), 0)
    for glyph in glyphs:
        atlas.paste(glyph.mask, (glyph.atlas_x, glyph.atlas_y))
    return atlas


def write_assets(output: Path, atlas: Image.Image, glyphs: list[Glyph], strikes: tuple[int, ...], fonts: tuple[Path, ...], codepoints: tuple[int, ...], support: tuple[set[int], ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage.bin").write_bytes(atlas.tobytes())
    with (output / "glyphs.bin").open("wb") as stream:
        for glyph in glyphs:
            stream.write(struct.pack("<IIIIIIIfff", glyph.font, glyph.strike, glyph.codepoint,
                                     glyph.atlas_x, glyph.atlas_y, glyph.mask.width,
                                     glyph.mask.height, glyph.bearing_x, glyph.bearing_y, glyph.advance))
    with (output / "support.bin").open("wb") as stream:
        for font_support in support:
            stream.write(bytes(cp in font_support for cp in codepoints))
    metadata = {
        "format": "oa_text_atlas_v1", "width": atlas.width, "height": atlas.height,
        "font_count": 3, "strikes": strikes, "codepoints": codepoints,
        "glyph_record_bytes": 40, "pillow": PILLOW_VERSION,
        "freetype": features.version_module("freetype2"),
        "font_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in fonts],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sans-font", type=Path, required=True)
    parser.add_argument("--sans-semibold-font", type=Path, required=True)
    parser.add_argument("--mono-font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--padding", type=int, default=1)
    args = parser.parse_args()
    fonts = (args.sans_font, args.mono_font, args.sans_semibold_font)
    support = tuple(unicode_cmap(path) for path in fonts)
    candidates = {cp for first, last in CANDIDATE_RANGES for cp in range(first, last + 1)}
    codepoints = tuple(sorted(candidates & set().union(*support)))
    glyphs = []
    for font_id, (font, supported) in enumerate(zip(fonts, support)):
        glyphs.extend(rasterize(font_id, font, DEFAULT_STRIKES, codepoints, supported))
    write_assets(args.output, pack(glyphs, args.width, args.padding), glyphs,
                 DEFAULT_STRIKES, fonts, codepoints, support)


if __name__ == "__main__":
    main()
