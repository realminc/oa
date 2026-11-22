use std::fmt;

/// Append infallible formatted text to an owned string.
pub(crate) fn push_format(output: &mut String, arguments: fmt::Arguments<'_>) {
	use fmt::Write as _;
	output
		.write_fmt(arguments)
		.expect("formatting into a String cannot fail");
}

/// Append one correctly escaped JSON string value.
pub(crate) fn push_json_string(output: &mut String, value: &str) {
	const HEX: &[u8; 16] = b"0123456789abcdef";
	output.push('"');
	for character in value.chars() {
		match character {
			'"' => output.push_str("\\\""),
			'\\' => output.push_str("\\\\"),
			'\u{0008}' => output.push_str("\\b"),
			'\u{000c}' => output.push_str("\\f"),
			'\n' => output.push_str("\\n"),
			'\r' => output.push_str("\\r"),
			'\t' => output.push_str("\\t"),
			'\u{0000}'..='\u{001f}' => {
				let value = character as u32;
				output.push_str("\\u00");
				output.push(HEX[((value >> 4) & 0xf) as usize] as char);
				output.push(HEX[(value & 0xf) as usize] as char);
			}
			_ => output.push(character),
		}
	}
	output.push('"');
}
