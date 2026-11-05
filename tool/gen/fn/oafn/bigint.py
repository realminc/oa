"""BigInt schema validation and complete emitters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_physical_write, validate_unique
from .util import banner, format_rust, require, rust_const_name

BIGINT_EXTENDED = {
	"div_rem_wide", "div_rem_narrow", "div_rem_mixed",
	"add_scalar", "subtract_scalar", "multiply_scalar", "compare_scalar", "div_rem_scalar",
	"from_u32", "from_u64", "to_u32", "to_u64", "reduce_montgomery",
}


BIGINT_NAMES = {
	"add", "subtract", "multiply_low", "multiply_wide", "bit_and", "bit_or",
	"bit_xor", "compare", "div_rem", "shift_left", "shift_right",
	"multiply_high", "square_wide", "bit_not", "leading_zeros",
}


BIGINT_MODULAR_HELPERS = {
	"reduce_mod": "bigint_reduce", "add_mod": "bigint_add_mod",
	"subtract_mod": "bigint_subtract_mod", "multiply_mod": "bigint_multiply_mod",
	"square_mod": "bigint_multiply_mod", "inverse_mod": "bigint_inverse_mod",
	"pow_mod": "bigint_pow_mod", "to_montgomery": "bigint_to_montgomery",
	"from_montgomery": "bigint_from_montgomery",
	"montgomery_multiply": "bigint_montgomery_multiply",
	"barrett_multiply": "bigint_barrett_multiply",
}


BIGINT_MODULAR_UNARY = {"reduce_mod", "square_mod", "inverse_mod", "to_montgomery", "from_montgomery"}


BIGINT_PREPARED = {"setup_barrett", "reduce_barrett"}


BIGINT_UNARY = BIGINT_MODULAR_UNARY | {"square_wide", "bit_not", "leading_zeros"}
BIGINT_NAMES |= BIGINT_MODULAR_HELPERS.keys() | BIGINT_EXTENDED | BIGINT_PREPARED


def bigint_extended_inputs(name: str) -> list[str]:
	return ['left', 'right'] if name.startswith('div_rem_') and name != 'div_rem_scalar' else ['left', 'modulus'] if name == 'reduce_montgomery' else ['left']


def bigint_extended_parameters(name: str) -> list[str]:
	return ['quotient_words'] if name == 'div_rem_mixed' else ['output_words'] if name.startswith('from_') else ['scalar'] if name.endswith('_scalar') else []


def bigint_extended_results(name: str) -> list[dict[str, str]]:
	if name.startswith('div_rem_'):
		return [{'name': 'quotient', 'extent': 'wide' if name == 'div_rem_wide' else 'quotient' if name == 'div_rem_mixed' else 'row'}, {'name': 'remainder', 'extent': 'scalar' if name == 'div_rem_scalar' else 'row'}, {'name': 'status', 'extent': 'scalar'}]
	return [{'name': 'output', 'extent': 'scalar' if name in {'to_u32', 'compare_scalar'} else 'pair' if name == 'to_u64' else 'row'}] + ([{'name': 'status', 'extent': 'scalar'}] if name.startswith('to_') or name == 'reduce_montgomery' else [])


def validate_bigint_extended(kernel: dict[str, Any], contracts: list[dict[str, Any]]) -> None:
	name = kernel['name']
	inputs, parameters, outputs = bigint_extended_inputs(name), bigint_extended_parameters(name), bigint_extended_results(name)
	fields = [n + '_index' for n in inputs] + [o['name'] + '_index' for o in outputs] + ['count', 'words', 'input_words'] + parameters + (['right_count'] if len(inputs) == 2 else [])
	attributes = [['width_bits', 'unsigned_integer']] + [[n if n != 'words' else 'output_words', 'unsigned_integer'] for n in parameters]
	expected = {'name': name, 'input_kinds': ['matrix'] * len(inputs), 'output_kinds': ['matrix'] * len(outputs), 'shape_rule': 'explicit', 'dtype_rule': 'u32', 'attributes': attributes}
	writes = [{'binding': len(inputs) + i, 'domain': 'rows', 'partition': 'exclusive_per_invocation', 'extent': 'one_scalar' if o['extent'] == 'scalar' else 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'} for i, o in enumerate(outputs)]
	require(kernel['outputs'] == outputs and kernel['push_fields'] == [[f, 'uint32'] for f in fields]
		and kernel['kernel_id'] == 'CryptographyBigInt' + ''.join(w.title() for w in name.split('_')) + 'U32'
		and kernel['source'] == f'src/slang/crypto/bigint/{name}.gen.slang'
		and kernel['semantic_operation'] == name and kernel['workgroup_size'] == [32, 1, 1], 'BigInt extended ABI differs')
	validate_physical_write(kernel['physical_write'], 'BigInt', required=True)
	require(kernel['physical_write'] == {'writes': writes, 'workspace': 'none'}, 'BigInt extended ownership differs')
	require(sum(c == expected for c in contracts) == 1, 'BigInt extended contract differs')


def bigint_extended_output(kernel: dict[str, Any]) -> tuple[str, str]:
	name = kernel['name']
	inputs, parameters, results = bigint_extended_inputs(name), bigint_extended_parameters(name), kernel['outputs']
	division = name in {'div_rem_wide', 'div_rem_narrow', 'div_rem_mixed'}
	conversion = name.startswith('from_')
	# Each generated operation owns validation, allocation, ABI and recording.
	dimensions = '''	let [rows, input_words] = match left.shape() {
		[rows, words] => [*rows, *words],
		_ => return Err(Error::invalid_argument(format!("{operation} requires rank-two U32 rows"))),
	};'''
	validation = 'left.dtype() != DType::U32 || !(1..=128).contains(&input_words)'
	if division or name == 'reduce_montgomery':
		right = 'right' if division else 'modulus'
		dimensions += f'''
	let words = match {right}.shape() {{
		[_, words] => *words,
		_ => return Err(Error::invalid_argument(format!("{{operation}} requires rank-two divisors"))),
	}};'''
		validation = f'left.dtype() != DType::U32 || !(1..=256).contains(&input_words) || !(1..=128).contains(&words)\n\t\t|| {right}.dtype() != DType::U32 || ({right}.shape()[0] != 1 && {right}.shape()[0] != rows)\n\t\t|| !left.engine_handle().same_as({right}.engine_handle())'
		if name != 'div_rem_mixed':
			validation += '\n\t\t|| input_words != words * 2'
		else:
			validation += '\n\t\t|| !(1..=256).contains(&quotient_words)'
	elif conversion:
		dimensions += '\n\tlet words = output_words as usize;'
		validation = f'left.dtype() != DType::U32 || input_words != {2 if name == "from_u64" else 1} || !({2 if name == "from_u64" else 1}..=128).contains(&words)'
	else:
		dimensions += '\n\tlet words = input_words;'
	widths = {'row': 'words', 'wide': 'input_words', 'quotient': 'quotient_words as usize', 'scalar': '1', 'pair': '2'}
	allocation = []
	for result in results:
		n, extent = result['name'], result['extent']; w = widths[extent]
		shape = 'vec![rows]' if extent == 'scalar' else f'vec![rows, {w}]'
		allocation.append(f'''	let {n}_elements = rows.checked_mul({w})
		.ok_or_else(|| Error::invalid_argument(format!("{{operation}} output extent overflows usize")))?;
	let {n}_bytes = {n}_elements.checked_mul(4)
		.ok_or_else(|| Error::invalid_argument(format!("{{operation}} output byte extent overflows usize")))?;
	u32::try_from({n}_bytes).map_err(|_| Error::invalid_argument(format!("{{operation}} output byte extent exceeds u32")))?;
	let {n} = Matrix::allocate_output(engine, {shape}, {n}_elements, DType::U32)?;''')
	result = '(' + ', '.join(o['name'] for o in results) + ')' if len(results) > 1 else 'output'
	return_type = '(' + ', '.join(['Matrix'] * len(results)) + ')' if len(results) > 1 else 'Matrix'
	behavior = ('Division returns (quotient, remainder, status): 1 success, 0 zero divisor, 2 quotient overflow.\n/// Invalid rows have completely zero outputs. The divisor may be one shared row.\n/// Mixed division accepts independent dividend (1..=256), divisor (1..=128)\n/// and quotient (1..=256) word counts; wide/narrow division takes a double-width dividend.' if division else
		'Returns (output, status); 1 means exact conversion, 0 means overflow with zero output.\n/// U64 values use two little-endian U32 words per row, without shaderInt64.' if name.startswith('to_') else
		'Zero-extends one-word U32 or two-word U64 rows into the requested word width.' if conversion else
		'Returns (output, status). Computes T * R^-1 mod m for a double-width T.\n/// R = 2^(32*words); odd nonzero m is required. Invalid rows are zero.' if name == 'reduce_montgomery' else
		'Returns (quotient, remainder, status). Scalar zero yields status 0 and zero outputs; valid status is 1.' if name == 'div_rem_scalar' else
		'Returns U32 order codes: 0 less, 1 equal, 2 greater.' if name == 'compare_scalar' else
		'Unsigned scalar arithmetic wraps within the complete declared limb width.')
	bindings = '\n'.join([f'\t\t\tBufferBinding::read({n}.storage()),' for n in inputs] + [f"\t\t\tBufferBinding::write({o['name']}.storage())," for o in results])
	push = '\n'.join([f'\t\t\tPushConstant::U32({n}' + (' as u32' if n == 'words' else '') + '),' for n in parameters] + ([f'\t\t\tPushConstant::U32({inputs[1]}.shape()[0] as u32),'] if len(inputs) == 2 else []))
	attrs = '\n'.join(f'\t\t\tOpAttribute::UnsignedInteger {{ name: "{n if n != "words" else "output_words"}".into(), value: {n} as u64 }},' for n in parameters)
	arguments = ', '.join([n + ': &Matrix' for n in inputs] + [n + ': u32' for n in parameters])
	function = bigint_template('bigint_extended.rs.in', dict(name=name, arguments=arguments, return_type=return_type,
		constant=rust_const_name(name), behavior=behavior, dimensions=dimensions, validation=validation,
		allocations='\n'.join(allocation), bindings=bindings, push=push, attributes=attrs, kernel_id=kernel['kernel_id'],
		inputs=', '.join(inputs), outputs=', '.join('&' + o['name'] for o in results), result=result))
	fields = '\n'.join(f'\tuint {f};' for f, _ in kernel['push_fields'])
	bounds = ['\tuint left_extent; storage_buffers[push.left_index].GetDimensions(left_extent);\n\tif (push.count > left_extent / (push.input_words * 4u)) return;']
	if len(inputs) == 2:
		n = inputs[1]
		bounds.append(f'\tuint right_extent; storage_buffers[push.{n}_index].GetDimensions(right_extent);\n\tif ((push.right_count != 1u && push.right_count != push.count) || push.right_count > right_extent / (push.words * 4u)) return;')
	shader_widths = {'row': 'push.words', 'wide': 'push.input_words', 'quotient': 'push.quotient_words', 'scalar': '1u', 'pair': '2u'}
	for o in results:
		n, w = o['name'], shader_widths[o['extent']]
		bounds.append(f'\tuint {n}_extent; uint {n}_row_bytes = {w} * 4u;\n\tstorage_buffers[push.{n}_index].GetDimensions({n}_extent);\n\tif (push.count > {n}_extent / {n}_row_bytes) return;')
	guards = 'push.input_words == 0u || push.input_words > 256u'
	if division or name == 'reduce_montgomery':
		guards += ' || push.words == 0u || push.words > 128u'
		guards += ' || push.quotient_words == 0u || push.quotient_words > 256u' if name == 'div_rem_mixed' else ' || push.input_words != push.words * 2u'
	elif conversion:
		guards += f' || push.input_words != {2 if name == "from_u64" else 1}u || push.words < {2 if name == "from_u64" else 1}u || push.words > 128u'
	else:
		guards += ' || push.input_words != push.words || push.words > 128u'
	body = []
	if division:
		body.append('''	uint a[256], b[128], quotient[256], remainder[128];
	for (uint i = 0u; i < 256u; ++i) a[i] = 0u;
	bigint_zero(b);
	for (uint i = 0u; i < push.input_words; ++i)
		a[i] = storage_buffers[push.left_index].Load(row * push.input_words * 4u + i * 4u);
	uint right_base = (push.right_count == 1u ? 0u : row) * push.words * 4u;
	for (uint i = 0u; i < push.words; ++i)
		b[i] = storage_buffers[push.right_index].Load(right_base + i * 4u);
	uint status = uint(bigint_divide_mixed(a, b, push.input_words, push.words, quotient, remainder));''')
		qw = shader_widths[results[0]['extent']]
		body.append(f'''	uint quotient_words = {qw};
	uint overflow = 0u;
	for (uint i = quotient_words; i < push.input_words; ++i) overflow |= quotient[i];
	if (status != 0u && overflow != 0u) status = 2u;
	for (uint i = 0u; i < quotient_words; ++i)
		storage_buffers[push.quotient_index].Store(row * quotient_row_bytes + i * 4u, status == 1u ? quotient[i] : 0u);
	for (uint i = 0u; i < push.words; ++i)
		storage_buffers[push.remainder_index].Store(row * remainder_row_bytes + i * 4u, status == 1u ? remainder[i] : 0u);
	storage_buffers[push.status_index].Store(row * 4u, status);''')
	elif conversion:
		body.append('''	for (uint i = 0u; i < push.words; ++i) {
		uint value = i < push.input_words ? storage_buffers[push.left_index].Load(row * push.input_words * 4u + i * 4u) : 0u;
		storage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, value);
	}''')
	elif name.startswith('to_'):
		w = 1 if name == 'to_u32' else 2
		body.append(f'''	uint overflow = 0u;
	for (uint i = {w}u; i < push.words; ++i)
		overflow |= storage_buffers[push.left_index].Load(row * push.words * 4u + i * 4u);
	for (uint i = 0u; i < {w}u; ++i) {{
		uint value = i < push.words ? storage_buffers[push.left_index].Load(row * push.words * 4u + i * 4u) : 0u;
		storage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, overflow == 0u ? value : 0u);
	}}
	storage_buffers[push.status_index].Store(row * 4u, uint(overflow == 0u));''')
	elif name == 'reduce_montgomery':
		body.append('''	uint a[256], modulus[128], result[128];
	for (uint i = 0u; i < 256u; ++i) a[i] = 0u;
	bigint_zero(modulus);
	for (uint i = 0u; i < push.input_words; ++i)
		a[i] = storage_buffers[push.left_index].Load(row * push.input_words * 4u + i * 4u);
	uint right_base = (push.right_count == 1u ? 0u : row) * push.words * 4u;
	for (uint i = 0u; i < push.words; ++i)
		modulus[i] = storage_buffers[push.modulus_index].Load(right_base + i * 4u);
	bool valid = bigint_montgomery_reduce_wide(a, modulus, push.words, result);
	for (uint i = 0u; i < push.words; ++i)
		storage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, result[i]);
	storage_buffers[push.status_index].Store(row * 4u, uint(valid));''')
	else:
		body.append('''	uint a[128], b[128], result[128];
	bigint_zero(a); bigint_zero(b); b[0] = push.scalar;
	for (uint i = 0u; i < push.words; ++i)
		a[i] = storage_buffers[push.left_index].Load(row * push.words * 4u + i * 4u);''')
		if name == 'compare_scalar':
			body.append('\tstorage_buffers[push.output_index].Store(row * 4u, uint(bigint_compare(a, b, push.words) + 1));')
		elif name == 'div_rem_scalar':
			body.append('''	uint remainder[128];
	bool valid = bigint_divide(a, b, push.words, result, remainder);
	for (uint i = 0u; i < push.words; ++i)
		storage_buffers[push.quotient_index].Store(row * quotient_row_bytes + i * 4u, valid ? result[i] : 0u);
	storage_buffers[push.remainder_index].Store(row * 4u, valid ? remainder[0] : 0u);
	storage_buffers[push.status_index].Store(row * 4u, uint(valid));''')
		elif name == 'multiply_scalar':
			body.append('''	uint carry = 0u;
	for (uint i = 0u; i < push.words; ++i) {
		uint2 product = mul32_wide(a[i], push.scalar);
		uint low = product.x + carry;
		carry = product.y + uint(low < product.x);
		storage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, low);
	}''')
		else:
			helper = 'bigint_add' if name == 'add_scalar' else 'bigint_subtract'
			body.append(f'''\t{helper}(a, b, push.words, result);
\tfor (uint i = 0u; i < push.words; ++i)
\t\tstorage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, result[i]);''')
	shader = bigint_template('bigint_extended.slang.in', dict(fields=fields, name=name, guards=guards, bounds='\n'.join(bounds), body='\n'.join(body)))
	return function, shader


def bigint_barrett_output(kernel: dict[str, Any]) -> tuple[str, str]:
	name = kernel['name']; setup = name == 'setup_barrett'
	if setup:
		function = '''/// Prepare a device-owned public Barrett modulus snapshot and reciprocal.
///
/// # Errors
/// Rejects dtype, rank, width, byte extents, allocation or recording failures.
pub fn setup_barrett(modulus: &Matrix) -> Result<BarrettParameters> {
	let contract = crate::core::operation::crypto::bigint::SETUP_BARRETT;
	let [rows, words] = match modulus.shape() {
		[rows, words] => [*rows, *words],
		_ => return Err(Error::invalid_argument("setup_barrett requires rank-two modulus rows")),
	};
	if modulus.dtype() != DType::U32 || !(1..=128).contains(&words) {
		return Err(Error::invalid_argument("setup_barrett requires U32 modulus rows with 1..=128 words"));
	}
	let count = u32::try_from(rows).map_err(|_| Error::invalid_argument("setup_barrett batch exceeds u32"))?;
	let encoded_words = words * 2 + 3;
	let elements = rows.checked_mul(encoded_words).ok_or_else(|| Error::invalid_argument("setup_barrett extent overflows usize"))?;
	let bytes = elements.checked_mul(4).ok_or_else(|| Error::invalid_argument("setup_barrett bytes overflow usize"))?;
	u32::try_from(bytes).map_err(|_| Error::invalid_argument("setup_barrett bytes exceed u32"))?;
	let engine = modulus.engine_handle();
	let encoded = Matrix::allocate_output(engine, vec![rows, encoded_words], elements, DType::U32)?;
	let status = Matrix::allocate_output(engine, vec![rows], rows, DType::U32)?;
	if rows != 0 {
		let kernel = KernelId::CryptographyBigIntSetupBarrettU32;
		let buffers = [BufferBinding::read(modulus.storage()), BufferBinding::write(encoded.storage()), BufferBinding::write(status.storage())];
		let push_constants = [PushConstant::U32(count), PushConstant::U32(words as u32)];
		let attributes = [OpAttribute::UnsignedInteger { name: "width_bits".into(), value: (words * 32) as u64 }];
		engine.record_semantic(
			ComputeDispatch { kernel, buffers: &buffers, push_constants: &push_constants, workgroups: kernel.linear_workgroups(count), indirect: None },
			SemanticDispatch { contract, inputs: &[modulus], outputs: &[&encoded, &status], attributes: &attributes },
		)?;
	}
	Ok(BarrettParameters { encoded, status, words })
}'''
	else:
		function = '''/// Reduce double-width public rows using a prepared immutable Barrett snapshot.
///
/// Returns `(output, status)`: one for success, zero for a zero modulus, two
/// when T is outside the significant-width Barrett bound T < 2^(64*k).
/// Invalid rows produce zero. The prepared modulus may have one shared row.
///
/// # Errors
/// Rejects dtype, shape, ownership, byte extents, allocation or recording failures.
pub fn reduce_barrett(left: &Matrix, parameters: &BarrettParameters) -> Result<(Matrix, Matrix)> {
	let contract = crate::core::operation::crypto::bigint::REDUCE_BARRETT;
	let words = parameters.words;
	let [rows, input_words] = match left.shape() {
		[rows, input_words] => [*rows, *input_words],
		_ => return Err(Error::invalid_argument("reduce_barrett requires rank-two double-width rows")),
	};
	let prepared = &parameters.encoded;
	if left.dtype() != DType::U32 || input_words != words * 2
		|| (prepared.shape()[0] != 1 && prepared.shape()[0] != rows)
		|| !left.engine_handle().same_as(prepared.engine_handle()) {
		return Err(Error::invalid_argument("reduce_barrett requires matching Engine and modulus width/batch"));
	}
	let count = u32::try_from(rows).map_err(|_| Error::invalid_argument("reduce_barrett batch exceeds u32"))?;
	let input_bytes = left.num_elements().checked_mul(4).ok_or_else(|| Error::invalid_argument("reduce_barrett input bytes overflow usize"))?;
	u32::try_from(input_bytes).map_err(|_| Error::invalid_argument("reduce_barrett input bytes exceed u32"))?;
	let elements = rows.checked_mul(words).ok_or_else(|| Error::invalid_argument("reduce_barrett output extent overflows usize"))?;
	let engine = left.engine_handle();
	let output = Matrix::allocate_output(engine, vec![rows, words], elements, DType::U32)?;
	let status = Matrix::allocate_output(engine, vec![rows], rows, DType::U32)?;
	if rows != 0 {
		let kernel = KernelId::CryptographyBigIntReduceBarrettU32;
		let buffers = [BufferBinding::read(left.storage()), BufferBinding::read(prepared.storage()), BufferBinding::write(output.storage()), BufferBinding::write(status.storage())];
		let push_constants = [PushConstant::U32(count), PushConstant::U32(words as u32), PushConstant::U32(prepared.shape()[0] as u32)];
		let attributes = [OpAttribute::UnsignedInteger { name: "width_bits".into(), value: (words * 32) as u64 }];
		engine.record_semantic(
			ComputeDispatch { kernel, buffers: &buffers, push_constants: &push_constants, workgroups: kernel.linear_workgroups(count), indirect: None },
			SemanticDispatch { contract, inputs: &[left, prepared], outputs: &[&output, &status], attributes: &attributes },
		)?;
	}
	Ok((output, status))
}'''
	fields = '\n'.join(f'\tuint {f};' for f, _ in kernel['push_fields'])
	shader = f'''// Prepared public Barrett data is an immutable typed snapshot (see Rust API).
import attributes;
import bigint_words;
import bigint_barrett;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{name}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	uint row = tid.x;
	if (row >= push.count || push.words == 0u || push.words > 128u) return;
	uint row_bytes = push.words * 4u;
	uint prepared_bytes = (push.words * 2u + 3u) * 4u;
	uint status_extent; storage_buffers[push.status_index].GetDimensions(status_extent);
	if (push.count > status_extent / 4u) return;
'''
	if setup:
		shader += '''	uint modulus_extent, encoded_extent;
	storage_buffers[push.modulus_index].GetDimensions(modulus_extent);
	storage_buffers[push.encoded_index].GetDimensions(encoded_extent);
	if (push.count > modulus_extent / row_bytes || push.count > encoded_extent / prepared_bytes) return;
	uint modulus[128], mu[130], k;
	bigint_zero(modulus);
	for (uint i = 0u; i < push.words; ++i)
		modulus[i] = storage_buffers[push.modulus_index].Load(row * row_bytes + i * 4u);
	bool valid = bigint_barrett_reciprocal(modulus, push.words, mu, k);
	for (uint i = 0u; i < push.words; ++i)
		storage_buffers[push.encoded_index].Store(row * prepared_bytes + i * 4u, modulus[i]);
	for (uint i = 0u; i < push.words + 2u; ++i)
		storage_buffers[push.encoded_index].Store(row * prepared_bytes + (i + push.words) * 4u, mu[i]);
	storage_buffers[push.encoded_index].Store(row * prepared_bytes + (push.words * 2u + 2u) * 4u, k);
	storage_buffers[push.status_index].Store(row * 4u, uint(valid));
}\n'''
	else:
		shader += '''	uint left_extent, encoded_extent, output_extent;
	storage_buffers[push.left_index].GetDimensions(left_extent);
	storage_buffers[push.encoded_index].GetDimensions(encoded_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.count > left_extent / (row_bytes * 2u) || push.count > output_extent / row_bytes
		|| (push.prepared_count != 1u && push.prepared_count != push.count)
		|| push.prepared_count > encoded_extent / prepared_bytes) return;
	uint base = (push.prepared_count == 1u ? 0u : row) * prepared_bytes;
	uint modulus[128], mu[130], input[256], result[128];
	bigint_zero(modulus); bigint_zero(result);
	for (uint i = 0u; i < 130u; ++i) mu[i] = 0u;
	for (uint i = 0u; i < 256u; ++i) input[i] = 0u;
	for (uint i = 0u; i < push.words; ++i)
		modulus[i] = storage_buffers[push.encoded_index].Load(base + i * 4u);
	for (uint i = 0u; i < push.words + 2u; ++i)
		mu[i] = storage_buffers[push.encoded_index].Load(base + (i + push.words) * 4u);
	uint k = storage_buffers[push.encoded_index].Load(base + (push.words * 2u + 2u) * 4u);
	uint overflow = 0u;
	for (uint i = 0u; i < push.words * 2u; ++i) {
		input[i] = storage_buffers[push.left_index].Load(row * row_bytes * 2u + i * 4u);
		if (i >= k * 2u) overflow |= input[i];
	}
	uint status = k == 0u ? 0u : (k > push.words || overflow != 0u ? 2u : 1u);
	if (status == 1u) bigint_barrett_reduce(input, modulus, mu, k, result);
	for (uint i = 0u; i < push.words; ++i)
		storage_buffers[push.output_index].Store(row * row_bytes + i * 4u, result[i]);
	storage_buffers[push.status_index].Store(row * 4u, status);
}\n'''
	return function, shader


def validate_bigint_prepared(kernel: dict[str, Any], contracts: list[dict[str, Any]]) -> None:
	setup = kernel['name'] == 'setup_barrett'
	inputs = ['modulus'] if setup else ['left', 'encoded']
	outputs = [{'name': 'encoded', 'extent': 'prepared'}, {'name': 'status', 'extent': 'scalar'}] if setup else [{'name': 'output', 'extent': 'row'}, {'name': 'status', 'extent': 'scalar'}]
	fields = [n + '_index' for n in inputs] + [o['name'] + '_index' for o in outputs] + ['count', 'words'] + ([] if setup else ['prepared_count'])
	writes = [{'binding': len(inputs) + i, 'domain': 'rows', 'partition': 'exclusive_per_invocation', 'extent': 'one_scalar' if o['extent'] == 'scalar' else 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'} for i, o in enumerate(outputs)]
	expected = {'name': kernel['name'], 'input_kinds': ['matrix'] * len(inputs), 'output_kinds': ['matrix', 'matrix'], 'shape_rule': 'explicit', 'dtype_rule': 'u32', 'attributes': [['width_bits', 'unsigned_integer']]}
	require(kernel['outputs'] == outputs and kernel['push_fields'] == [[n, 'uint32'] for n in fields] and kernel['workgroup_size'] == [32, 1, 1]
		and kernel['kernel_id'] == 'CryptographyBigInt' + ''.join(w.title() for w in kernel['name'].split('_')) + 'U32'
		and kernel['source'] == f"src/slang/crypto/bigint/{kernel['name']}.gen.slang" and kernel['semantic_operation'] == kernel['name'], 'Prepared Barrett ABI differs')
	validate_physical_write(kernel['physical_write'], 'prepared Barrett', required=True)
	require(kernel['physical_write'] == {'writes': writes, 'workspace': 'none'} and sum(c == expected for c in contracts) == 1, 'Prepared Barrett ownership or contract differs')


def validate_bigint_schema(schema: dict[str, Any]) -> None:
	require(set(schema) <= {"schema_version", "family", "domain", "dtype", "limb_layout", "contracts", "kernels", "signature_records"} and {"schema_version", "family", "domain", "dtype", "limb_layout", "contracts", "kernels"} <= set(schema), "BigInt schema fields differ")
	require(type(schema["schema_version"]) is int and schema["schema_version"] == 1 and schema["family"] == "cryptography_bigint"
		and schema["domain"] == "crypto::bigint" and schema["dtype"] == "u32", "BigInt schema identity differs")
	layout = schema["limb_layout"]
	require(layout == {"word_bits": 32, "min_words": 1, "max_words": 128, "order": "little_endian"}
		and all(type(layout.get(k)) is int for k in ("word_bits", "min_words", "max_words")), "BigInt limb layout differs")
	kernels, contracts = schema["kernels"], schema["contracts"]
	cooperative = {k["name"]: k for k in kernels if "provider_of" in k}
	require(isinstance(kernels, list) and len(kernels) == len(BIGINT_NAMES) + len(cooperative)
		and isinstance(contracts, list) and len(contracts) == len(BIGINT_NAMES), "BigInt operation inventory differs")
	require(set(cooperative) == {f"{n}_cooperative" for n in ("add", "subtract", "multiply_wide", "square_wide")}, "BigInt cooperative provider inventory differs")
	names, ids = set(), set()
	for kernel in kernels:
		provider = "provider_of" in kernel
		fields_set = {"name", "stable_id", "kernel_id", "source", "workgroup_size", "push_fields", "semantic_operation", "outputs", "physical_write"}
		if provider:
			fields_set.add("provider_of")
		require(isinstance(kernel, dict) and set(kernel) == fields_set, "BigInt kernel fields differ")
		name = kernel["name"]
		if provider:
			base = kernel["provider_of"]
			require(name == f"{base}_cooperative" and base in BIGINT_NAMES and kernel["semantic_operation"] == base
				and kernel["source"] == f"src/slang/crypto/bigint/{name}.gen.slang"
				and kernel["kernel_id"] == "CryptographyBigInt" + ''.join(w.title() for w in name.split('_')) + "U32"
				and kernel["push_fields"] == [[f, "uint32"] for f in ["left_index", "right_index", "output_index", "count", "grid_width", "words"]]
				and kernel["outputs"] == [{"name": "output", "extent": "wide" if base in {"multiply_wide", "square_wide"} else "row"}]
				and kernel["workgroup_size"] == [32, 1, 1]
				and kernel["physical_write"] == {"writes": [{"binding": 2, "domain": "rows", "partition": "exclusive_per_workgroup", "extent": "row_width", "collision": "exclusive", "tail": "bounds_checked"}], "workspace": "none"}, "BigInt cooperative ABI differs")
		else:
			require(isinstance(name, str) and name in BIGINT_NAMES, "BigInt operation is unknown")
		validate_unique(name, names, "BigInt name")
		require(type(kernel["stable_id"]) is int and 0 < kernel["stable_id"] <= 65535, "BigInt stable ID is invalid")
		validate_unique(kernel["stable_id"], ids, "BigInt stable ID")
		if provider:
			continue
		if name in BIGINT_PREPARED:
			validate_bigint_prepared(kernel, contracts)
			continue
		if name in BIGINT_EXTENDED:
			validate_bigint_extended(kernel, contracts)
			continue
		shift, modular = name.startswith("shift_"), name in BIGINT_MODULAR_HELPERS
		unary = shift or name in BIGINT_UNARY
		outputs = [{"name": "output", "extent": "row"}, {"name": "status", "extent": "scalar"}] if modular else [{"name": "quotient", "extent": "row"}, {"name": "remainder", "extent": "row"}, {"name": "status", "extent": "scalar"}] if name == "div_rem" else [{"name": "output", "extent": "wide" if name in {"multiply_wide", "square_wide"} else "scalar" if name in {"compare", "leading_zeros"} else "row"}]
		fields = ["left_index"] + ([] if unary else ["right_index"]) + (["modulus_index"] if modular else []) + [o["name"] + "_index" for o in outputs] + ["count", "words"] + (["shift"] if shift else ["modulus_count"] if modular else [])
		require(kernel["kernel_id"] == "CryptographyBigInt" + ''.join(w.title() for w in name.split('_')) + "U32"
			and kernel["source"] == f"src/slang/crypto/bigint/{name}.gen.slang"
			and kernel["semantic_operation"] == name and kernel["outputs"] == outputs
			and kernel["push_fields"] == [[f, "uint32"] for f in fields], "BigInt ABI differs")
		wg = kernel["workgroup_size"]
		require(isinstance(wg, list) and len(wg) == 3 and all(type(n) is int and n > 0 for n in wg)
			and wg[1:] == [1, 1] and wg[0] <= 1024, "BigInt workgroup geometry differs")
		input_count = (1 if unary else 2) + int(modular)
		writes = [{"binding": input_count + i, "domain": "rows", "partition": "exclusive_per_invocation", "extent": "one_scalar" if o["extent"] == "scalar" else "row_width", "collision": "exclusive", "tail": "bounds_checked"} for i, o in enumerate(outputs)]
		validate_physical_write(kernel["physical_write"], "BigInt", required=True)
		require(all(type(w["binding"]) is int for w in kernel["physical_write"]["writes"]), "BigInt output binding must be an integer")
		require(kernel["physical_write"] == {"writes": writes, "workspace": "none"}, "BigInt output ownership differs")
		expected = {"name": name, "input_kinds": ["matrix"] * input_count, "output_kinds": ["matrix"] * len(outputs), "shape_rule": "explicit", "dtype_rule": "u32", "attributes": [["width_bits", "unsigned_integer"]] + ([["shift", "unsigned_integer"]] if shift else [])}
		require(sum(c == expected for c in contracts) == 1, "BigInt semantic contract differs")


def bigint_template(name: str, values: dict[str, str]) -> str:
	path = FN_ROOT / "template/crypto" / name
	body = path.read_text(encoding="utf-8")
	require(set(re.findall(r"\{\{([^{}]+)\}\}", body)) == set(values), "BigInt template substitution differs")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda m: values[m[1]], body)


def generate_bigint_cooperative(kernel: dict[str, Any], mark: str) -> str:
	name = kernel['name']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	if name in ('multiply_wide_cooperative', 'square_wide_cooperative'):
		shared = 'groupshared uint colsum[256 * 3];'
		body = '''	uint words2 = push.words * 2u;
	for (uint i = lane; i < words2 * 3u; i += 32u) colsum[i] = 0u;
	GroupMemoryBarrierWithGroupSync();
	for (uint k = lane; k < words2; k += 32u) {
		uint2 sum = uint2(0u, 0u);
		uint extra = 0u;
		uint first = k >= push.words ? k - push.words + 1u : 0u;
		uint last = min(k, push.words - 1u);
		[loop]
		for (uint i = first; i <= last; ++i) {
			uint a = storage_buffers[push.left_index].Load(input_base + i * 4u);
			uint b = storage_buffers[push.right_index].Load(input_base + (k - i) * 4u);
			uint2 product = mul32_wide(a, b);
			uint low = sum.x + product.x;
			uint carry_low = uint(low < sum.x);
			sum.x = low;
			uint high = sum.y + product.y;
			uint carry = uint(high < sum.y);
			sum.y = high + carry_low;
			carry += uint(sum.y < carry_low);
			extra += carry;
		}
		colsum[k * 3u] = sum.x;
		colsum[k * 3u + 1u] = sum.y;
		colsum[k * 3u + 2u] = extra;
	}
	GroupMemoryBarrierWithGroupSync();
	if (lane == 0u) {
		uint carry_low = 0u, carry_high = 0u;
		for (uint k = 0u; k < words2; ++k) {
			uint low = colsum[k * 3u] + carry_low;
			uint into_mid = uint(low < colsum[k * 3u]);
			uint mid = colsum[k * 3u + 1u] + carry_high;
			uint extra = uint(mid < colsum[k * 3u + 1u]);
			mid += into_mid;
			extra += uint(mid < into_mid);
			uint high = colsum[k * 3u + 2u] + extra;
			storage_buffers[push.output_index].Store(row * (push.words * 8u) + k * 4u, low);
			carry_low = mid;
			carry_high = high;
		}
	}'''
	elif name == 'add_cooperative':
		shared = 'groupshared uint lane_carry[32];\ngroupshared uint lane_propagate[32];'
		body = '''	uint chunk = (push.words + 31u) / 32u;
	uint lane_base = lane * chunk;
	uint generate = 0u, propagate = 1u;
	[loop]
	for (uint j = 0u; j < chunk; ++j) {
		uint i = lane_base + j;
		uint a = 0u, b = 0u;
		if (i < push.words) {
			a = storage_buffers[push.left_index].Load(input_base + i * 4u);
			b = storage_buffers[push.right_index].Load(input_base + i * 4u);
		}
		uint s0 = a + b + generate;
		generate = uint(s0 < a || (generate == 1u && s0 == a));
		uint s1 = a + b + propagate;
		propagate = uint(s1 < a || (propagate == 1u && s1 == a));
	}
	lane_carry[lane] = generate;
	lane_propagate[lane] = propagate;
	GroupMemoryBarrierWithGroupSync();
	if (lane == 0u) {
		uint carry = 0u;
		for (uint l = 0u; l < 32u; ++l) {
			uint next = carry == 0u ? lane_carry[l] : lane_propagate[l];
			lane_carry[l] = carry;
			carry = next;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	uint carry = lane_carry[lane];
	[loop]
	for (uint j = 0u; j < chunk; ++j) {
		uint i = lane_base + j;
		if (i < push.words) {
			uint a = storage_buffers[push.left_index].Load(input_base + i * 4u);
			uint b = storage_buffers[push.right_index].Load(input_base + i * 4u);
			uint sum = a + b + carry;
			carry = uint(sum < a || (carry == 1u && sum == a));
			storage_buffers[push.output_index].Store(input_base + i * 4u, sum);
		}
	}'''
	else:
		shared = 'groupshared uint lane_borrow[32];\ngroupshared uint lane_propagate[32];'
		body = '''	uint chunk = (push.words + 31u) / 32u;
	uint lane_base = lane * chunk;
	uint generate = 0u, propagate = 1u;
	[loop]
	for (uint j = 0u; j < chunk; ++j) {
		uint i = lane_base + j;
		uint a = 0u, b = 0u;
		if (i < push.words) {
			a = storage_buffers[push.left_index].Load(input_base + i * 4u);
			b = storage_buffers[push.right_index].Load(input_base + i * 4u);
		}
		generate = uint(a < b || (generate == 1u && a == b));
		propagate = uint(a < b || (propagate == 1u && a == b));
	}
	lane_borrow[lane] = generate;
	lane_propagate[lane] = propagate;
	GroupMemoryBarrierWithGroupSync();
	if (lane == 0u) {
		uint borrow = 0u;
		for (uint l = 0u; l < 32u; ++l) {
			uint next = borrow == 0u ? lane_borrow[l] : lane_propagate[l];
			lane_borrow[l] = borrow;
			borrow = next;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	uint borrow = lane_borrow[lane];
	[loop]
	for (uint j = 0u; j < chunk; ++j) {
		uint i = lane_base + j;
		if (i < push.words) {
			uint a = storage_buffers[push.left_index].Load(input_base + i * 4u);
			uint b = storage_buffers[push.right_index].Load(input_base + i * 4u);
			uint difference = a - b - borrow;
			borrow = uint(a < b || (borrow == 1u && a == b));
			storage_buffers[push.output_index].Store(input_base + i * 4u, difference);
		}
	}'''
	return mark + f'''// Cooperative-lane batch provider; one workgroup owns each batch row.
// Exact carry/borrow semantics match the per-invocation baseline.
import attributes;
import bigint_words;
import wide_u32;

struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
{shared}
[kernel_name("{name}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {{
	uint row = group.x + group.y * push.grid_width;
	// Every early exit is uniform across the workgroup, before any barrier.
	if (row >= push.count || push.words == 0u || push.words > 128u) return;
	uint input_row_bytes = push.words * 4u;
	uint left_extent, right_extent, output_extent;
	storage_buffers[push.left_index].GetDimensions(left_extent);
	storage_buffers[push.right_index].GetDimensions(right_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	uint output_row_bytes = {'input_row_bytes * 2u' if name in ('multiply_wide_cooperative', 'square_wide_cooperative') else 'input_row_bytes'};
	if (push.count > left_extent / input_row_bytes || push.count > right_extent / input_row_bytes
		|| push.count > output_extent / output_row_bytes) return;
	uint input_base = row * input_row_bytes;
{body}
}}\n'''


def bigint_outputs(root: Path, schema: dict[str, Any], schema_hash: str) -> dict[Path, str]:
	mark = banner(schema_hash, "//", "crypto_bigint.json")
	api = mark + 'use crate::{DType, Error, Matrix, OpAttribute, Result, runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch}};\n\n'
	python = mark + 'use pyo3::prelude::*;\nuse crate::{error::python_error, matrix::PythonMatrix};\n\n'
	stub = banner(schema_hash, "#", "crypto_bigint.json") + 'from .._native import Matrix\n\n'
	facade = banner(schema_hash, "#", "crypto_bigint.json") + '"""Public-data fixed-width device arithmetic."""\nfrom .._native import (\n'
	api += bigint_template('barrett_types.rs.in', {}) + '\n'
	python += '#[pyclass(name = "BarrettParameters", unsendable)]\nstruct PythonBarrettParameters { inner: oa::crypto::bigint::BarrettParameters }\n#[pymethods]\nimpl PythonBarrettParameters {\n\t#[getter]\n\tfn words(&self) -> usize { self.inner.words() }\n\t#[getter]\n\tfn status(&self) -> PythonMatrix { PythonMatrix::wrap(self.inner.status().clone()) }\n}\n\n'
	stub += 'class BarrettParameters:\n    @property\n    def words(self) -> int: ...\n    @property\n    def status(self) -> Matrix: ...\n\n'
	facade += '    BarrettParameters,\n'
	outputs, names = {}, []
	providers = {k['provider_of']: k for k in schema['kernels'] if 'provider_of' in k}
	for kernel in schema["kernels"]:
		name, results = kernel["name"], kernel["outputs"]
		if "provider_of" in kernel:
			outputs[root / kernel['source']] = generate_bigint_cooperative(kernel, mark)
			continue
		shift, div = name.startswith("shift_"), name == "div_rem"
		modular = name in BIGINT_MODULAR_HELPERS
		unary = shift or name in BIGINT_UNARY
		multiple = len(results) > 1
		names.append(name)
		if name in BIGINT_PREPARED:
			function, shader = bigint_barrett_output(kernel)
			api += function + '\n\n'
			outputs[root / kernel['source']] = mark + shader
			if name == 'setup_barrett':
				python += '#[pyfunction]\nfn crypto_bigint_setup_barrett(modulus: &PythonMatrix) -> PyResult<PythonBarrettParameters> {\n\tlet inner = oa::crypto::bigint::setup_barrett(&modulus.inner).map_err(python_error)?;\n\tOk(PythonBarrettParameters { inner })\n}\n\n'
				stub += 'def crypto_bigint_setup_barrett(modulus: Matrix) -> BarrettParameters: ...\n'
			else:
				python += '#[pyfunction]\nfn crypto_bigint_reduce_barrett(left: &PythonMatrix, parameters: &PythonBarrettParameters) -> PyResult<(PythonMatrix, PythonMatrix)> {\n\tlet (output, status) = oa::crypto::bigint::reduce_barrett(&left.inner, &parameters.inner).map_err(python_error)?;\n\tOk((PythonMatrix::wrap(output), PythonMatrix::wrap(status)))\n}\n\n'
				stub += 'def crypto_bigint_reduce_barrett(left: Matrix, parameters: BarrettParameters) -> tuple[Matrix, Matrix]: ...\n'
			facade += f'    crypto_bigint_{name} as bigint_{name},\n'
			continue
		input_names = ['left'] + ([] if unary else ['right']) + (['modulus'] if modular else [])
		if name in BIGINT_EXTENDED:
			input_names = bigint_extended_inputs(name)
		parameters = bigint_extended_parameters(name) if name in BIGINT_EXTENDED else (["shift"] if shift else [])
		arguments = ', '.join([n + ': &Matrix' for n in input_names] + [n + ': u32' for n in parameters])
		allocation = []
		for out in results:
			key, extent = out["name"], out["extent"]
			width = 'words * 2' if extent == 'wide' else '1' if extent == 'scalar' else 'words'
			shape = 'vec![rows]' if extent == 'scalar' else f'vec![rows, {width}]'
			allocation.append(f'''\tlet {key}_elements = rows.checked_mul({width})
\t\t.ok_or_else(|| Error::invalid_argument(format!("{{operation}} output extent overflows usize")))?;
\tlet {key}_bytes = {key}_elements.checked_mul(4)
\t\t.ok_or_else(|| Error::invalid_argument(format!("{{operation}} output byte extent overflows usize")))?;
\tu32::try_from({key}_bytes)
\t\t.map_err(|_| Error::invalid_argument(format!("{{operation}} output byte extent exceeds u32")))?;
\tlet {key} = Matrix::allocate_output(engine, {shape}, {key}_elements, DType::U32)?;''')
		result = '(' + ', '.join(o['name'] for o in results) + ')' if multiple else 'output'
		return_type = '(' + ', '.join(['Matrix'] * len(results)) + ')' if multiple else 'Matrix'
		summary = f"Compute batched unsigned limb {name.replace('_', ' ')}."
		behavior = ('Returns `(quotient, remainder, status)`; status is U32 `[batch]`.\n/// A zero divisor produces status 0, quotient 0 and remainder equal to the dividend.\n/// Otherwise status is 1 and division is exact.' if div else
			'Returns U32 `[batch]` order codes: 0 less, 1 equal, 2 greater.' if name == 'compare' else
			'Returns exact double-width rows (up to 8192 output bits).' if name in {'multiply_wide', 'square_wide'} else
			'Returns the upper half of the exact double-width product, with all low-half carries retained.' if name == 'multiply_high' else
			'Returns U32 `[batch]` leading-zero counts in the complete operand width; zero returns width_bits.' if name == 'leading_zeros' else
			'Returns the complement within the complete fixed operand width.' if name == 'bit_not' else
			'Shifts at or beyond the operand width produce zero; discarded bits are lost.' if shift else
			'Returns same-width rows; arithmetic overflow/underflow wraps modulo 2^width.')
		if modular:
			behavior = 'Returns `(output, status)` with U32 `[batch]` status flags.\n/// `modulus` is U32 `[1, words]` or `[batch, words]`, with matching Engine ownership.\n/// A zero modulus returns status 0 and zero output; valid rows return status 1.'
			if name == 'inverse_mod':
				behavior += '\n/// Non-invertible values also return status 0 and zero output. Odd/even and composite moduli are accepted.'
			elif name in {'to_montgomery', 'from_montgomery', 'montgomery_multiply'}:
				behavior += '\n/// Montgomery operations require an odd modulus; even moduli return status 0 and zero output.\n/// R = 2^(32*words). Conversion multiplies by R or R^-1; multiplication computes a*b*R^-1 modulo m.'
			elif name == 'pow_mod':
				behavior += '\n/// `right` is the unsigned exponent; zero exponents yield 1 modulo m, including 0^0.'
		validation = '' if unary else '\n\t\t|| right.dtype() != DType::U32 || right.shape() != left.shape()\n\t\t|| !left.engine_handle().same_as(right.engine_handle())'
		if modular:
			validation += '\n\t\t|| modulus.dtype() != DType::U32 || modulus.shape().len() != 2\n\t\t|| modulus.shape()[1] != words\n\t\t|| (modulus.shape()[0] != 1 && modulus.shape()[0] != rows)\n\t\t|| !left.engine_handle().same_as(modulus.engine_handle())'
		if name in BIGINT_EXTENDED:
			function, shader = bigint_extended_output(kernel)
			api += function + "\n"
			outputs[root/kernel["source"]] = mark + shader
		else:
			bindings = '\n'.join([f'\t\t\tBufferBinding::read({n}.storage()),' for n in input_names] + [f"\t\t\tBufferBinding::write({o['name']}.storage())," for o in results])
			inner = f'''		let kernel = KernelId::{kernel['kernel_id']};
		let buffers = [
{bindings}
		];
		let push_constants = [
			PushConstant::U32(count),
			PushConstant::U32(words as u32),
{chr(9) * 3 + 'PushConstant::U32(shift),' if shift else chr(9) * 3 + 'PushConstant::U32(modulus.shape()[0] as u32),' if modular else ''}
		];
		let inputs = [{', '.join(input_names)}];
		let outputs = [{', '.join('&' + o['name'] for o in results)}];
		let attributes = [
			OpAttribute::UnsignedInteger {{ name: "width_bits".into(), value: (words * 32) as u64 }},
{chr(9) * 3 + 'OpAttribute::UnsignedInteger { name: "shift".into(), value: u64::from(shift) },' if shift else ''}
		];
		engine.record_semantic(
			ComputeDispatch {{
				kernel, buffers: &buffers, push_constants: &push_constants,
				workgroups: kernel.linear_workgroups(count),
				indirect: None,
			}},
			SemanticDispatch {{ contract, inputs: &inputs, outputs: &outputs, attributes: &attributes }},
		)?;'''
			provider = providers.get(name)
			if provider is None:
				dispatch = '	if rows != 0 {\n' + inner + '\n\t}'
			else:
				coop_bindings = '\n'.join([f'\t\t\t\tBufferBinding::read({n}.storage()),' for n in (['left', 'left'] if name == 'square_wide' else input_names)] + [f"\t\t\t\tBufferBinding::write({o['name']}.storage())," for o in results])
				indented = '\n'.join('\t' + line if line else line for line in inner.split('\n'))
				dispatch = f'''	if rows != 0 {{
		// Small batches underutilize per-invocation scheduling; the cooperative
		// provider owns one workgroup per row. Routing is private lowering policy.
		if count <= 32u32 {{
			let kernel = KernelId::{provider['kernel_id']};
			let buffers = [
{coop_bindings}
			];
			let groups = engine.linear_dispatch_grid(count, 1)?;
			let push_constants = [
				PushConstant::U32(count),
				PushConstant::U32(groups[0]),
				PushConstant::U32(words as u32),
			];
			let inputs = [{', '.join(input_names)}];
			let outputs = [{', '.join('&' + o['name'] for o in results)}];
			let attributes = [
				OpAttribute::UnsignedInteger {{ name: "width_bits".into(), value: (words * 32) as u64 }},
			];
			engine.record_semantic(
				ComputeDispatch {{
					kernel, buffers: &buffers, push_constants: &push_constants,
					workgroups: groups,
					indirect: None,
				}},
				SemanticDispatch {{ contract, inputs: &inputs, outputs: &outputs, attributes: &attributes }},
			)?;
		}} else {{
{indented}
		}}
	}}'''
			api += bigint_template('bigint.rs.in', dict(summary=summary, behavior=behavior, name=name, arguments=arguments, return_type=return_type,
				constant=rust_const_name(name), validation=validation, allocations='\n'.join(allocation), dispatch=dispatch, result=result)) + '\n'
			fields = '\n'.join(f'\tuint {field};' for field, _ in kernel['push_fields'])
			bounds = ['\tuint left_extent;\n\tstorage_buffers[push.left_index].GetDimensions(left_extent);\n\tif (push.count > left_extent / input_row_bytes) return;']
			if not unary:
				bounds.append('\tuint right_extent;\n\tstorage_buffers[push.right_index].GetDimensions(right_extent);\n\tif (push.count > right_extent / input_row_bytes) return;')
			if modular:
				bounds.append('\tuint modulus_extent;\n\tstorage_buffers[push.modulus_index].GetDimensions(modulus_extent);\n\tif ((push.modulus_count != 1u && push.modulus_count != push.count)\n\t\t|| push.modulus_count > modulus_extent / input_row_bytes) return;')
			for o in results:
				key = o['name']
				stride = 'input_row_bytes * 2u' if o['extent'] == 'wide' else '4u' if o['extent'] == 'scalar' else 'input_row_bytes'
				bounds.append(f'\tuint {key}_extent;\n\tuint {key}_row_bytes = {stride};\n\tstorage_buffers[push.{key}_index].GetDimensions({key}_extent);\n\tif (push.count > {key}_extent / {key}_row_bytes) return;')
			body = []
			if modular:
				body.append('''\tuint a[128], modulus[128];
	\tbigint_zero(a); bigint_zero(modulus);
	\tuint modulus_base = (push.modulus_count == 1u ? 0u : row) * input_row_bytes;
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {
	\t\ta[i] = storage_buffers[push.left_index].Load(input_base + i * 4u);
	\t\tmodulus[i] = storage_buffers[push.modulus_index].Load(modulus_base + i * 4u);
	\t}''')
				if not unary:
					body.append('''\tuint b[128];
	\tbigint_zero(b);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i)
	\t\tb[i] = storage_buffers[push.right_index].Load(input_base + i * 4u);''')
				args = 'a, a, modulus' if name == 'square_mod' else 'a, modulus' if unary else 'a, b, modulus'
				body.append(f'''\tuint result[128];
	\tbool valid = {BIGINT_MODULAR_HELPERS[name]}({args}, push.words, result);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i)
	\t\tstorage_buffers[push.output_index].Store(input_base + i * 4u, result[i]);
	\tstorage_buffers[push.status_index].Store(row * 4u, uint(valid));''')
			elif name == 'bit_not':
				body.append('''\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {
	\t\tuint address = input_base + i * 4u;
	\t\tstorage_buffers[push.output_index].Store(address, ~storage_buffers[push.left_index].Load(address));
	\t}''')
			elif name == 'leading_zeros':
				body.append('''\tuint a[128];
	\tbigint_zero(a);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i)
	\t\ta[i] = storage_buffers[push.left_index].Load(input_base + i * 4u);
	\tstorage_buffers[push.output_index].Store(row * 4u, bigint_leading_zeros(a, push.words));''')
			elif name.startswith('bit_'):
				operator = {'bit_and':'&', 'bit_or':'|', 'bit_xor':'^'}[name]
				body.append(f'''\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {{
	\t\tuint address = input_base + i * 4u;
	\t\tuint value = storage_buffers[push.left_index].Load(address) {operator} storage_buffers[push.right_index].Load(address);
	\t\tstorage_buffers[push.output_index].Store(address, value);
	\t}}''')
			elif shift:
				body.append('''\tuint word_shift = push.shift / 32u;
	\tuint bit_shift = push.shift % 32u;
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {
	\t\tuint value = 0u;''')
				if name == 'shift_left':
					body.append('''\t\tif (push.shift < push.words * 32u && i >= word_shift) {
	\t\t\tuint source = i - word_shift;
	\t\t\tvalue = storage_buffers[push.left_index].Load(input_base + source * 4u) << bit_shift;
	\t\t\tif (bit_shift != 0u && source != 0u)
	\t\t\t\tvalue |= storage_buffers[push.left_index].Load(input_base + (source - 1u) * 4u) >> (32u - bit_shift);
	\t\t}''')
				else:
					body.append('''\t\tif (push.shift < push.words * 32u && i + word_shift < push.words) {
	\t\t\tuint source = i + word_shift;
	\t\t\tvalue = storage_buffers[push.left_index].Load(input_base + source * 4u) >> bit_shift;
	\t\t\tif (bit_shift != 0u && source + 1u < push.words)
	\t\t\t\tvalue |= storage_buffers[push.left_index].Load(input_base + (source + 1u) * 4u) << (32u - bit_shift);
	\t\t}''')
				body.append('\t\tstorage_buffers[push.output_index].Store(input_base + i * 4u, value);\n\t}')
			elif name == 'square_wide':
				body.append('''\tuint a[128];
	\tbigint_zero(a);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i)
	\t\ta[i] = storage_buffers[push.left_index].Load(input_base + i * 4u);
	\tuint product[256];
	\tbigint_multiply(a, a, push.words, push.words * 2u, product);
	\t[loop]
	\tfor (uint i = 0u; i < push.words * 2u; ++i)
	\t\tstorage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, product[i]);''')
			else:
				right = 'push.right_index'
				body.append(f'''\tuint a[128], b[128];
	\tbigint_zero(a); bigint_zero(b);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {{
	\t\ta[i] = storage_buffers[push.left_index].Load(input_base + i * 4u);
	\t\tb[i] = storage_buffers[{right}].Load(input_base + i * 4u);
	\t}}''')
				if name == 'compare':
					body.append('\tstorage_buffers[push.output_index].Store(row * 4u, uint(bigint_compare(a, b, push.words) + 1));')
				elif div:
					body.append('''\tuint quotient[128], remainder[128];
	\tbool valid = bigint_divide(a, b, push.words, quotient, remainder);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i) {
	\t\tstorage_buffers[push.quotient_index].Store(input_base + i * 4u, quotient[i]);
	\t\tstorage_buffers[push.remainder_index].Store(input_base + i * 4u, remainder[i]);
	\t}
	\tstorage_buffers[push.status_index].Store(row * 4u, uint(valid));''')
				elif name.startswith('multiply_') or name == 'square_wide':
					wide = name in {'multiply_wide', 'square_wide'}
					width = 'push.words * 2u' if wide else 'push.words'
					product_width = 'push.words * 2u' if wide or name == 'multiply_high' else 'push.words'
					index = 'i + push.words' if name == 'multiply_high' else 'i'
					body.append(f'''\tuint product[256];
	\tbigint_multiply(a, b, push.words, {product_width}, product);
	\t[loop]
	\tfor (uint i = 0u; i < {width}; ++i)
	\t\tstorage_buffers[push.output_index].Store(row * output_row_bytes + i * 4u, product[{index}]);''')
				else:
					helper = 'bigint_add' if name == 'add' else 'bigint_subtract'
					body.append(f'''\tuint result[128];
	\t{helper}(a, b, push.words, result);
	\t[loop]
	\tfor (uint i = 0u; i < push.words; ++i)
	\t\tstorage_buffers[push.output_index].Store(input_base + i * 4u, result[i]);''')
			imports = 'import bigint_words;' + ('\nimport bigint_barrett;' if name == 'barrett_multiply' else '\nimport bigint_modular;' if modular else '')
			outputs[root/kernel['source']] = mark + bigint_template('bigint.slang.in', dict(imports=imports, fields=fields, name=name, workgroup=', '.join(map(str,kernel['workgroup_size'])), bounds='\n'.join(bounds), body='\n'.join(body)))
		if name in BIGINT_EXTENDED:
			behavior = 'Mixed/wide/scalar division statuses: 1 success, 0 zero divisor, 2 quotient overflow; invalid rows are zero. Conversion to U32/U64 returns exact-value status 1, or overflow status 0 with a zero result.' if name.startswith('div_rem') or name in {'to_u32', 'to_u64'} else 'Public fixed-width scalar/word conversion or arithmetic; no secret execution contract.'
		py_args = ', '.join([n + ': &PythonMatrix' for n in input_names] + [n + ': u32' for n in parameters])
		py_return = '(' + ', '.join(['PythonMatrix'] * len(results)) + ')' if multiple else 'PythonMatrix'
		call = ', '.join(['&'+n+'.inner' for n in input_names] + parameters)
		py_result = 'let '+result+' = value;\n\tOk(('+', '.join('PythonMatrix::wrap('+o['name']+')' for o in results)+'))' if multiple else 'Ok(PythonMatrix::wrap(value))'
		python += f'/// {summary}\n///\n/// Public U32 limb rows, 32..=4096 bits; no secret-execution contract.\n/// {behavior}\n#[pyfunction]\nfn crypto_bigint_{name}({py_args}) -> PyResult<{py_return}> {{\n\tlet value = oa::crypto::bigint::{name}({call}).map_err(python_error)?;\n\t{py_result}\n}}\n\n'
		stub_args = ', '.join([n+': Matrix' for n in input_names] + [n+': int' for n in parameters])
		stub_return = 'tuple['+', '.join(['Matrix']*len(results))+']' if multiple else 'Matrix'
		stub += f'def crypto_bigint_{name}({stub_args}) -> {stub_return}: ...\n'
		facade += f'    crypto_bigint_{name} as bigint_{name},\n'
	python += 'pub(super) fn register(module: &Bound<\'_, PyModule>) -> PyResult<()> {\n\tmodule.add_class::<PythonBarrettParameters>()?;\n' + ''.join(f'\tmodule.add_function(wrap_pyfunction!(crypto_bigint_{n}, module)?)?;\n' for n in names) + '\tOk(())\n}\n'
	facade += ')\n\n__all__ = '+repr(['BarrettParameters'] + ['bigint_'+n for n in names])+'\n'
	outputs.update({root/'src/rs/crypto/bigint/arithmetic.gen.rs':format_rust(api,root), root/'src/rs/crypto/bigint/exports.gen.rs':format_rust(mark+'pub use arithmetic::{BarrettParameters, '+', '.join(names)+'};\n',root), root/'src/py/pyo3/crypto/bigint.gen.rs':format_rust(python,root), root/'src/py/oa/_gen/crypto_bigint.py':facade, root/'src/py/oa/_gen/crypto_bigint_native.pyi':stub})
	return outputs
