"""UI style token schema emitted as ``src/rs/lib/ui/style.gen.rs``.

``ui_style.json`` is the single owner of the ``UiStyle`` token set: the
ordered color/metric token list, the shared six-color ``data_palette`` and
named presets. The first preset is the base and must resolve every token;
later presets declare ``extends`` plus only their overrides so derived
themes (``viewer_dark``, future editor profiles) stay token resolutions
instead of second palette authors.

The emitter writes fully resolved ``pub(crate) fn ui_style_<name>()`` preset
constructors; handwritten ``UiStyle`` factory methods delegate to them.
"""

from __future__ import annotations

import re
from typing import Any

from .model import validate_unique
from .util import IDENTIFIER, banner, require

_COLOR_HEX = re.compile(r"^0x[0-9A-Fa-f]{8}$")

_FIELD_ORDER_NOTE = (
	"color tokens first, then data_palette, then metric tokens — "
	"matching the UiStyle field order in src/rs/lib/ui/type.rs"
)


def _is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_color(value: Any, where: str) -> None:
	if isinstance(value, str):
		require(_COLOR_HEX.fullmatch(value) is not None,
			f"{where} color hex must be 0xRRGGBBAA")
		return
	require(isinstance(value, list) and len(value) == 4
		and all(_is_number(channel) and 0.0 <= channel <= 1.0 for channel in value),
		f"{where} color must be 0xRRGGBBAA or [r, g, b, a] in 0..=1")


def validate_ui_style_schema(schema: dict[str, Any]) -> None:
	"""Validate the ``ui_style`` token preset schema."""
	require(schema.get("schema_version") == 1, "UiStyle schema_version must be 1")
	domain = schema.get("domain")
	require(isinstance(domain, str) and IDENTIFIER.fullmatch(domain), "UiStyle domain must be an identifier")
	require(schema.get("family") == f"{domain}_style", f"UiStyle family must be {domain}_style")
	color_tokens = schema.get("color_tokens")
	require(isinstance(color_tokens, list) and color_tokens and all(
		isinstance(token, str) and IDENTIFIER.fullmatch(token) for token in color_tokens),
		"UiStyle color_tokens must be a nonempty identifier array")
	require(len(set(color_tokens)) == len(color_tokens), "UiStyle color_tokens must be unique")
	metric_tokens = schema.get("metric_tokens")
	require(isinstance(metric_tokens, list) and metric_tokens and all(
		isinstance(token, str) and IDENTIFIER.fullmatch(token) for token in metric_tokens),
		"UiStyle metric_tokens must be a nonempty identifier array")
	require(len(set(metric_tokens)) == len(metric_tokens), "UiStyle metric_tokens must be unique")
	require(not set(color_tokens) & set(metric_tokens), "UiStyle token names must not overlap")
	require("data_palette" not in set(color_tokens) | set(metric_tokens),
		"UiStyle data_palette is reserved")
	palette = schema.get("data_palette")
	require(isinstance(palette, list) and len(palette) == 6,
		"UiStyle data_palette must hold exactly 6 colors")
	for index, color in enumerate(palette):
		_validate_color(color, f"UiStyle data_palette[{index}]")
	all_tokens = set(color_tokens) | set(metric_tokens)
	presets = schema.get("presets")
	require(isinstance(presets, list) and presets, "UiStyle presets must be a nonempty array")
	names: set[str] = set()
	for index, preset in enumerate(presets):
		where = f"UiStyle preset[{index}]"
		require(isinstance(preset, dict) and set(preset) <= {"name", "extends", "doc", "values"},
			f"{where} record is incomplete or unknown")
		name = preset["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{where} name is invalid")
		validate_unique(name, names, "UiStyle preset")
		require(isinstance(preset["doc"], str) and "\n" not in preset["doc"],
			f"{where} {name} docs are invalid")
		extends = preset.get("extends")
		if index == 0:
			require(extends is None, "UiStyle base preset must not extend another preset")
		else:
			require(isinstance(extends, str) and extends in names,
				f"{where} {name} extends must reference an earlier preset")
		values = preset.get("values")
		require(isinstance(values, dict) and values, f"{where} {name} values must be a nonempty object")
		unknown = set(values) - all_tokens
		require(not unknown, f"{where} {name} unknown tokens: {sorted(unknown)}")
		if index == 0:
			require(set(values) == all_tokens,
				f"UiStyle base preset {name} must resolve every token")
		for token, value in values.items():
			if token in set(metric_tokens):
				require(_is_number(value) and value >= 0,
					f"{where} {name}.{token} metric must be a non-negative number")
			else:
				_validate_color(value, f"{where} {name}.{token}")


def _rust_color(value: Any) -> str:
	if isinstance(value, str):
		return f"Color::new_from_u32({value})"
	r, g, b, a = (repr(float(channel)) for channel in value)
	return f"Color::new({r}, {g}, {b}, {a})"


def _resolve(preset: dict[str, Any], by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
	resolved: dict[str, Any] = {}
	if "extends" in preset:
		resolved.update(_resolve(by_name[preset["extends"]], by_name))
	resolved.update(preset["values"])
	return resolved


def generate_ui_style(schema: dict[str, Any], schema_hash: str, schema_name: str) -> str:
	"""Emit ``style.gen.rs`` with one resolved preset constructor per preset."""
	text = banner(schema_hash, "//", schema_name)
	text += (
		"//! UiStyle token presets resolved from the schema. Token order: "
		+ _FIELD_ORDER_NOTE + "\n\n"
	)
	text += "use crate::Color;\nuse super::UiStyle;\n\n"
	text += "/// Shared six-color data series palette; alpha-zero artist styles consume it in order.\n"
	text += "pub(crate) const REALM_DATA_PALETTE: [Color; 6] = [\n"
	for color in schema["data_palette"]:
		text += f"\t{_rust_color(color)},\n"
	text += "];\n\n"
	by_name = {preset["name"]: preset for preset in schema["presets"]}
	for preset in schema["presets"]:
		resolved = _resolve(preset, by_name)
		doc_lines = "".join(f"/// {line}\n" for line in preset["doc"].split(". ") if line)
		if not doc_lines:
			doc_lines = f"/// {preset['name']} preset.\n"
		text += doc_lines
		text += f"///\n/// Token preset `{preset['name']}`"
		if "extends" in preset:
			text += f" (extends `{preset['extends']}`)"
		text += ".\n"
		text += f"pub(crate) fn ui_style_{preset['name']}() -> UiStyle {{\n\tUiStyle {{\n"
		for token in schema["color_tokens"]:
			text += f"\t\t{token}: {_rust_color(resolved[token])},\n"
		text += "\t\tdata_palette: REALM_DATA_PALETTE,\n"
		for token in schema["metric_tokens"]:
			text += f"\t\t{token}: {repr(float(resolved[token]))},\n"
		text += "\t}\n}\n\n"
	return text.rstrip("\n") + "\n"
