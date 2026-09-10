//! Native OA model-file (`.oam`) wire format.
//!
//! This module is deliberately host-only. It validates and owns persisted
//! bytes; `checkpoint` remains responsible for translating those bytes to and
//! from engine-owned values.

use std::{
	collections::HashSet,
	fs::{self, File},
	io::Write,
	path::{Path, PathBuf},
	sync::atomic::{AtomicU64, Ordering},
};

use crate::{DType, Error, Result};

pub(crate) const VERSION: u32 = 3;
const MIN_VERSION: u32 = 1;
const MAGIC: u32 = 0x004d_414f;
const PAGE_SIZE: usize = 4096;
const FILE_HEADER_SIZE: usize = 64;
const SECTION_HEADER_SIZE: usize = 64;
const CONFIG_SIZE: usize = 108;
const TENSOR_ENTRY_SIZE: usize = 216;
const OPTIMIZER_SIZE: usize = 68;
const PROGRESS_SIZE: usize = 104;
const MAX_SECTIONS: usize = 32;
const MAX_RANK: usize = 8;
const MAX_NAME: usize = 128;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum SectionKind {
	Config = 1,
	Weights = 2,
	State = 3,
	Optimizer = 4,
	Progress = 5,
	LegacyKernelCache = 6,
}

impl SectionKind {
	fn parse(value: u32) -> Result<Self> {
		match value {
			1 => Ok(Self::Config),
			2 => Ok(Self::Weights),
			3 => Ok(Self::State),
			4 => Ok(Self::Optimizer),
			5 => Ok(Self::Progress),
			6 => Ok(Self::LegacyKernelCache),
			_ => Err(corrupt("unknown section type")),
		}
	}

	const fn aligns_payload(self) -> bool {
		matches!(self, Self::Weights | Self::State)
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorEncoding {
	Dense,
	Q4,
	Q8,
}

impl TensorEncoding {
	const fn wire(self) -> u8 {
		match self {
			Self::Dense => 0,
			Self::Q4 => 1,
			Self::Q8 => 2,
		}
	}

	fn parse(value: u8) -> Result<Self> {
		match value {
			0 => Ok(Self::Dense),
			1 => Ok(Self::Q4),
			2 => Ok(Self::Q8),
			_ => Err(corrupt("unknown tensor encoding")),
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarType {
	F32,
	Bf16,
	F16,
	I8,
	I32,
	U8,
	F64,
	I16,
	I64,
	U16,
	U32,
	U64,
	Bool,
	Complex64,
	Complex128,
}

impl ScalarType {
	const fn wire(self) -> u8 {
		match self {
			Self::F32 => 0,
			Self::Bf16 => 1,
			Self::F16 => 2,
			Self::I8 => 3,
			Self::I32 => 4,
			Self::U8 => 5,
			Self::F64 => 6,
			Self::I16 => 7,
			Self::I64 => 8,
			Self::U16 => 9,
			Self::U32 => 10,
			Self::U64 => 11,
			Self::Bool => 12,
			Self::Complex64 => 13,
			Self::Complex128 => 14,
		}
	}

	fn parse(value: u8) -> Result<Self> {
		match value {
			0 => Ok(Self::F32),
			1 => Ok(Self::Bf16),
			2 => Ok(Self::F16),
			3 => Ok(Self::I8),
			4 => Ok(Self::I32),
			5 => Ok(Self::U8),
			6 => Ok(Self::F64),
			7 => Ok(Self::I16),
			8 => Ok(Self::I64),
			9 => Ok(Self::U16),
			10 => Ok(Self::U32),
			11 => Ok(Self::U64),
			12 => Ok(Self::Bool),
			13 => Ok(Self::Complex64),
			14 => Ok(Self::Complex128),
			_ => Err(corrupt("invalid scalar type")),
		}
	}

	const fn size_bytes(self) -> u64 {
		match self {
			Self::I8 | Self::U8 | Self::Bool => 1,
			Self::Bf16 | Self::F16 | Self::I16 | Self::U16 => 2,
			Self::F32 | Self::I32 | Self::U32 => 4,
			Self::F64 | Self::I64 | Self::U64 | Self::Complex64 => 8,
			Self::Complex128 => 16,
		}
	}
}

impl From<DType> for ScalarType {
	fn from(value: DType) -> Self {
		match value {
			DType::U8 => Self::U8,
			DType::F32 => Self::F32,
			DType::I32 => Self::I32,
			DType::U32 => Self::U32,
		}
	}
}

#[derive(Clone, Debug)]
pub(crate) struct Tensor {
	pub(crate) name: String,
	pub(crate) dtype: ScalarType,
	pub(crate) shape: Vec<u64>,
	pub(crate) encoding: TensorEncoding,
	pub(crate) block_size: u8,
	pub(crate) data: Vec<u8>,
}

impl Tensor {
	pub(crate) fn dense(
		name: impl Into<String>,
		dtype: DType,
		shape: Vec<u64>,
		data: Vec<u8>,
	) -> Result<Self> {
		let tensor = Self {
			name: name.into(),
			dtype: dtype.into(),
			shape,
			encoding: TensorEncoding::Dense,
			block_size: 0,
			data,
		};
		tensor.validate(true, VERSION, false)?;
		Ok(tensor)
	}

	fn validate(&self, weight: bool, version: u32, corrupt_input: bool) -> Result<()> {
		let invalid = |message| {
			if corrupt_input {
				corrupt(message)
			} else {
				Error::invalid_argument(format!("invalid .oam tensor: {message}"))
			}
		};
		if self.name.is_empty()
			|| self.name.as_bytes().contains(&0)
			|| self.name.len() >= MAX_NAME
			|| self.shape.len() > MAX_RANK
		{
			return Err(invalid("invalid name or rank"));
		}
		let elements = self
			.shape
			.iter()
			.try_fold(1_u64, |total, extent| total.checked_mul(*extent))
			.ok_or_else(|| invalid("element count overflow"))?;
		let expected = match self.encoding {
			TensorEncoding::Dense => {
				if self.block_size != 0 {
					return Err(invalid("dense tensor has a nonzero block size"));
				}
				elements
					.checked_mul(self.dtype.size_bytes())
					.ok_or_else(|| invalid("dense byte count overflow"))?
			}
			TensorEncoding::Q4 | TensorEncoding::Q8 => {
				if version < 3
					|| !weight || self.dtype != ScalarType::F32
					|| self.block_size != 32
					|| elements == 0
				{
					return Err(invalid("invalid quantized tensor layout"));
				}
				let blocks = elements.div_ceil(32);
				let payload = match self.encoding {
					TensorEncoding::Q4 => 16,
					TensorEncoding::Q8 => 32,
					TensorEncoding::Dense => unreachable!(),
				};
				blocks
					.checked_mul(payload + 4)
					.ok_or_else(|| invalid("quantized byte count overflow"))?
			}
		};
		if u64::try_from(self.data.len()).ok() != Some(expected) {
			return Err(invalid("byte count does not match shape and encoding"));
		}
		Ok(())
	}
}

#[derive(Clone, Debug)]
pub(crate) struct Config {
	pub(crate) architecture: String,
	pub(crate) config_version: u32,
	pub(crate) flags: u32,
	pub(crate) d_model: u32,
	pub(crate) n_layers: u32,
	pub(crate) d_vocab: u32,
	pub(crate) arch_config: Vec<u8>,
	pub(crate) weight_dtype: ScalarType,
	pub(crate) state_dtype: ScalarType,
	pub(crate) compute_dtype: ScalarType,
}

impl Default for Config {
	fn default() -> Self {
		Self {
			architecture: "RustModule".to_owned(),
			config_version: 1,
			flags: 0,
			d_model: 0,
			n_layers: 0,
			d_vocab: 256,
			arch_config: Vec::new(),
			weight_dtype: ScalarType::F32,
			state_dtype: ScalarType::F32,
			compute_dtype: ScalarType::F32,
		}
	}
}

#[derive(Clone, Debug)]
pub(crate) struct Optimizer {
	pub(crate) kind: String,
	pub(crate) learning_rate: f32,
	pub(crate) beta1: f32,
	pub(crate) beta2: f32,
	pub(crate) epsilon: f32,
	pub(crate) weight_decay: f32,
	pub(crate) step: i64,
	pub(crate) first_moment: Vec<f32>,
	pub(crate) second_moment: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct Progress {
	pub(crate) phase: u8,
	pub(crate) step: i64,
	pub(crate) bytes_seen: u64,
	pub(crate) env_steps: u64,
	pub(crate) learning_rate: f32,
	pub(crate) best_metric: f32,
	pub(crate) lower_is_better: bool,
	pub(crate) metric_name: String,
}

impl Default for Progress {
	fn default() -> Self {
		Self {
			phase: 0,
			step: 0,
			bytes_seen: 0,
			env_steps: 0,
			learning_rate: 3.0e-4,
			best_metric: 0.0,
			lower_is_better: true,
			metric_name: "loss".to_owned(),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelFile {
	pub(crate) config: Config,
	pub(crate) weights: Vec<Tensor>,
	pub(crate) state: Vec<Tensor>,
	pub(crate) optimizer: Option<Optimizer>,
	pub(crate) progress: Progress,
}

impl ModelFile {
	pub(crate) fn new() -> Self {
		Self::default()
	}

	pub(crate) fn save(&self, path: &Path) -> Result<()> {
		let bytes = self.encode()?;
		atomic_write(path, &bytes)
	}

	pub(crate) fn load(path: &Path) -> Result<Self> {
		let bytes = fs::read(path).map_err(|source| Error::io("read .oam", source))?;
		Self::decode(&bytes)
	}

	fn encode(&self) -> Result<Vec<u8>> {
		validate_name(&self.config.architecture, 32, "architecture")?;
		validate_name(&self.progress.metric_name, 32, "metric name")?;
		validate_tensors(&self.weights, true, VERSION, false)?;
		validate_tensors(&self.state, false, VERSION, false)?;

		let mut payloads = vec![(SectionKind::Config, encode_config(&self.config)?)];
		if !self.weights.is_empty() {
			payloads.push((SectionKind::Weights, encode_tensors(&self.weights)?));
		}
		if !self.state.is_empty() {
			payloads.push((SectionKind::State, encode_tensors(&self.state)?));
		}
		if let Some(optimizer) = &self.optimizer {
			payloads.push((SectionKind::Optimizer, encode_optimizer(optimizer)?));
		}
		payloads.push((SectionKind::Progress, encode_progress(&self.progress)?));

		let table_end = FILE_HEADER_SIZE
			.checked_add(
				payloads
					.len()
					.checked_mul(SECTION_HEADER_SIZE)
					.ok_or_else(|| Error::resource_exhausted(".oam section table size overflow"))?,
			)
			.ok_or_else(|| Error::resource_exhausted(".oam header size overflow"))?;
		let mut offset = page_align(table_end)?;
		let mut sections = Vec::with_capacity(payloads.len());
		for (kind, payload) in &payloads {
			let size = payload.len();
			sections.push(Section {
				kind: *kind,
				offset: u64::try_from(offset)
					.map_err(|_| Error::resource_exhausted(".oam offset exceeds u64"))?,
				size: u64::try_from(size)
					.map_err(|_| Error::resource_exhausted(".oam section exceeds u64"))?,
				checksum: hash(payload),
			});
			let occupied = if kind.aligns_payload() {
				page_align(size)?
			} else {
				size
			};
			offset = offset
				.checked_add(occupied)
				.ok_or_else(|| Error::resource_exhausted(".oam file size overflow"))?;
		}

		let total_size = u64::try_from(offset)
			.map_err(|_| Error::resource_exhausted(".oam file size exceeds u64"))?;
		let mut header = encode_file_header(payloads.len(), total_size, 0)?;
		let section_table = encode_section_table(&sections);
		let mut manifest = header.clone();
		manifest.extend_from_slice(&section_table);
		let checksum = hash(&manifest);
		header[24..32].copy_from_slice(&checksum.to_le_bytes());

		let mut bytes = vec![0_u8; offset];
		bytes[..FILE_HEADER_SIZE].copy_from_slice(&header);
		bytes[FILE_HEADER_SIZE..table_end].copy_from_slice(&section_table);
		for ((_, payload), section) in payloads.iter().zip(&sections) {
			let start = usize::try_from(section.offset)
				.map_err(|_| Error::resource_exhausted(".oam offset exceeds usize"))?;
			let end = start + payload.len();
			bytes[start..end].copy_from_slice(payload);
		}
		Ok(bytes)
	}

	fn decode(bytes: &[u8]) -> Result<Self> {
		if bytes.len() < FILE_HEADER_SIZE {
			return Err(corrupt("truncated file header"));
		}
		let mut header = Reader::new(&bytes[..FILE_HEADER_SIZE]);
		if header.u32()? != MAGIC {
			return Err(corrupt("invalid magic"));
		}
		let version = header.u32()?;
		if !(MIN_VERSION..=VERSION).contains(&version) {
			return Err(corrupt("unsupported format version"));
		}
		let count =
			usize::try_from(header.u32()?).map_err(|_| corrupt("section count exceeds usize"))?;
		if count == 0 || count > MAX_SECTIONS {
			return Err(corrupt("invalid section count"));
		}
		let _flags = header.u32()?;
		let total_size =
			usize::try_from(header.u64()?).map_err(|_| corrupt("file size exceeds usize"))?;
		let file_checksum = header.u64()?;
		if total_size != bytes.len() {
			return Err(corrupt("file size does not match header"));
		}
		let table_size = count
			.checked_mul(SECTION_HEADER_SIZE)
			.ok_or_else(|| corrupt("section table size overflow"))?;
		let table_end = FILE_HEADER_SIZE
			.checked_add(table_size)
			.ok_or_else(|| corrupt("section table size overflow"))?;
		let table = bytes
			.get(FILE_HEADER_SIZE..table_end)
			.ok_or_else(|| corrupt("truncated section table"))?;
		if version >= 2 {
			let mut normalized = bytes[..FILE_HEADER_SIZE].to_vec();
			normalized[24..32].fill(0);
			normalized.extend_from_slice(table);
			if hash(&normalized) != file_checksum {
				return Err(corrupt("file metadata checksum mismatch"));
			}
		}

		let data_start = page_align(table_end).map_err(|_| corrupt("header alignment overflow"))?;
		let mut sections = Vec::with_capacity(count);
		let mut seen = HashSet::new();
		let mut legacy_checksum = 0_u64;
		let mut table_reader = Reader::new(table);
		for _ in 0..count {
			let kind = SectionKind::parse(table_reader.u32()?)?;
			if !seen.insert(kind as u32) {
				return Err(corrupt("duplicate section type"));
			}
			if table_reader.u8()? != 0 {
				return Err(corrupt("unsupported section compression"));
			}
			table_reader.skip(3)?;
			let _flags = table_reader.u32()?;
			let offset = table_reader.u64()?;
			let size = table_reader.u64()?;
			if table_reader.u64()? != 0 {
				return Err(corrupt("unsupported section compression"));
			}
			let checksum = table_reader.u64()?;
			table_reader.skip(20)?;
			let start =
				usize::try_from(offset).map_err(|_| corrupt("section offset exceeds usize"))?;
			let size_usize =
				usize::try_from(size).map_err(|_| corrupt("section size exceeds usize"))?;
			let end = start
				.checked_add(size_usize)
				.ok_or_else(|| corrupt("section range overflow"))?;
			let payload = bytes
				.get(start..end)
				.filter(|_| start >= data_start)
				.ok_or_else(|| corrupt("section range is outside the file"))?;
			if hash(payload) != checksum {
				return Err(corrupt("section payload checksum mismatch"));
			}
			legacy_checksum ^= checksum;
			sections.push(Section {
				kind,
				offset,
				size,
				checksum,
			});
		}
		let mut ranges = sections
			.iter()
			.map(|section| (section.offset, section.offset + section.size))
			.collect::<Vec<_>>();
		ranges.sort_unstable_by_key(|range| range.0);
		if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
			return Err(corrupt("overlapping section ranges"));
		}
		if version == 1 && legacy_checksum != file_checksum {
			return Err(corrupt("legacy file checksum mismatch"));
		}
		if !seen.contains(&(SectionKind::Config as u32))
			|| !seen.contains(&(SectionKind::Progress as u32))
		{
			return Err(corrupt("required section is missing"));
		}

		let payload = |kind| -> Result<&[u8]> {
			let section = sections
				.iter()
				.find(|section| section.kind == kind)
				.ok_or_else(|| corrupt("required section is missing"))?;
			let start =
				usize::try_from(section.offset).map_err(|_| corrupt("offset exceeds usize"))?;
			let size = usize::try_from(section.size).map_err(|_| corrupt("size exceeds usize"))?;
			Ok(&bytes[start..start + size])
		};
		let config = decode_config(payload(SectionKind::Config)?)?;
		let weights = sections
			.iter()
			.find(|section| section.kind == SectionKind::Weights)
			.map(|_| {
				decode_tensors(
					payload(SectionKind::Weights).expect("located section"),
					true,
					version,
				)
			})
			.transpose()?
			.unwrap_or_default();
		let state = sections
			.iter()
			.find(|section| section.kind == SectionKind::State)
			.map(|_| {
				decode_tensors(
					payload(SectionKind::State).expect("located section"),
					false,
					version,
				)
			})
			.transpose()?
			.unwrap_or_default();
		let optimizer = sections
			.iter()
			.find(|section| section.kind == SectionKind::Optimizer)
			.map(|_| decode_optimizer(payload(SectionKind::Optimizer).expect("located section")))
			.transpose()?;
		let progress = decode_progress(payload(SectionKind::Progress)?)?;
		Ok(Self {
			config,
			weights,
			state,
			optimizer,
			progress,
		})
	}
}

#[derive(Clone, Copy)]
struct Section {
	kind: SectionKind,
	offset: u64,
	size: u64,
	checksum: u64,
}

fn encode_file_header(count: usize, total_size: u64, checksum: u64) -> Result<Vec<u8>> {
	let mut bytes = Vec::with_capacity(FILE_HEADER_SIZE);
	push_u32(&mut bytes, MAGIC);
	push_u32(&mut bytes, VERSION);
	push_u32(
		&mut bytes,
		u32::try_from(count).map_err(|_| Error::resource_exhausted("too many .oam sections"))?,
	);
	push_u32(&mut bytes, 0);
	push_u64(&mut bytes, total_size);
	push_u64(&mut bytes, checksum);
	bytes.resize(FILE_HEADER_SIZE, 0);
	Ok(bytes)
}

fn encode_section_table(sections: &[Section]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(sections.len() * SECTION_HEADER_SIZE);
	for section in sections {
		push_u32(&mut bytes, section.kind as u32);
		bytes.push(0);
		bytes.extend_from_slice(&[0; 3]);
		push_u32(&mut bytes, 0);
		push_u64(&mut bytes, section.offset);
		push_u64(&mut bytes, section.size);
		push_u64(&mut bytes, 0);
		push_u64(&mut bytes, section.checksum);
		bytes.extend_from_slice(&[0; 20]);
	}
	bytes
}

fn encode_config(config: &Config) -> Result<Vec<u8>> {
	let mut bytes = Vec::with_capacity(CONFIG_SIZE + config.arch_config.len());
	push_name(&mut bytes, &config.architecture, 32, "architecture")?;
	push_u32(&mut bytes, config.config_version);
	push_u32(&mut bytes, config.flags);
	push_u32(&mut bytes, config.d_model);
	push_u32(&mut bytes, config.n_layers);
	push_u32(&mut bytes, config.d_vocab);
	push_u32(
		&mut bytes,
		u32::try_from(config.arch_config.len())
			.map_err(|_| Error::resource_exhausted("architecture config exceeds u32"))?,
	);
	bytes.extend_from_slice(&[
		config.weight_dtype.wire(),
		config.state_dtype.wire(),
		config.compute_dtype.wire(),
		0,
	]);
	bytes.extend_from_slice(&[0; 48]);
	debug_assert_eq!(bytes.len(), CONFIG_SIZE);
	bytes.extend_from_slice(&config.arch_config);
	Ok(bytes)
}

fn decode_config(bytes: &[u8]) -> Result<Config> {
	if bytes.len() < CONFIG_SIZE {
		return Err(corrupt("invalid config section"));
	}
	let mut reader = Reader::new(bytes);
	let architecture = reader.name(32, "architecture")?;
	let config_version = reader.u32()?;
	let flags = reader.u32()?;
	let d_model = reader.u32()?;
	let n_layers = reader.u32()?;
	let d_vocab = reader.u32()?;
	let arch_size = usize::try_from(reader.u32()?)
		.map_err(|_| corrupt("architecture config size exceeds usize"))?;
	let weight_dtype = ScalarType::parse(reader.u8()?)?;
	let state_dtype = ScalarType::parse(reader.u8()?)?;
	let compute_dtype = ScalarType::parse(reader.u8()?)?;
	reader.skip(1 + 48)?;
	if CONFIG_SIZE.checked_add(arch_size) != Some(bytes.len()) {
		return Err(corrupt("architecture config size mismatch"));
	}
	Ok(Config {
		architecture,
		config_version,
		flags,
		d_model,
		n_layers,
		d_vocab,
		arch_config: reader.take(arch_size)?.to_vec(),
		weight_dtype,
		state_dtype,
		compute_dtype,
	})
}

fn encode_tensors(tensors: &[Tensor]) -> Result<Vec<u8>> {
	let index_size = 8_usize
		.checked_add(
			tensors
				.len()
				.checked_mul(TENSOR_ENTRY_SIZE)
				.ok_or_else(|| Error::resource_exhausted(".oam tensor index size overflow"))?,
		)
		.ok_or_else(|| Error::resource_exhausted(".oam tensor index size overflow"))?;
	let blob_size = tensors.iter().try_fold(0_usize, |total, tensor| {
		total
			.checked_add(tensor.data.len())
			.ok_or_else(|| Error::resource_exhausted(".oam tensor blob size overflow"))
	})?;
	let mut bytes = Vec::with_capacity(index_size + blob_size);
	push_u32(
		&mut bytes,
		u32::try_from(tensors.len())
			.map_err(|_| Error::resource_exhausted("too many .oam tensors"))?,
	);
	push_u32(&mut bytes, 0);
	let mut blob_offset = 0_u64;
	for tensor in tensors {
		push_name(&mut bytes, &tensor.name, MAX_NAME, "tensor name")?;
		push_u64(&mut bytes, blob_offset);
		push_u64(
			&mut bytes,
			u64::try_from(tensor.data.len())
				.map_err(|_| Error::resource_exhausted("tensor payload exceeds u64"))?,
		);
		bytes.push(tensor.dtype.wire());
		bytes.push(
			u8::try_from(tensor.shape.len())
				.map_err(|_| Error::invalid_argument("tensor rank exceeds u8"))?,
		);
		bytes.push(tensor.encoding.wire());
		bytes.push(tensor.block_size);
		bytes.extend_from_slice(&[0; 4]);
		for dimension in 0..MAX_RANK {
			push_u64(
				&mut bytes,
				tensor.shape.get(dimension).copied().unwrap_or(0),
			);
		}
		blob_offset = blob_offset
			.checked_add(u64::try_from(tensor.data.len()).unwrap_or(u64::MAX))
			.ok_or_else(|| Error::resource_exhausted("tensor blob offset overflow"))?;
	}
	for tensor in tensors {
		bytes.extend_from_slice(&tensor.data);
	}
	Ok(bytes)
}

fn decode_tensors(bytes: &[u8], weight: bool, version: u32) -> Result<Vec<Tensor>> {
	let mut reader = Reader::new(bytes);
	let count =
		usize::try_from(reader.u32()?).map_err(|_| corrupt("tensor count exceeds usize"))?;
	if reader.u32()? != 0 {
		return Err(corrupt("tensor index reserved field is nonzero"));
	}
	let index_size = count
		.checked_mul(TENSOR_ENTRY_SIZE)
		.and_then(|size| size.checked_add(8))
		.ok_or_else(|| corrupt("tensor index size overflow"))?;
	if index_size > bytes.len() {
		return Err(corrupt("truncated tensor index"));
	}
	let mut metadata = Vec::with_capacity(count);
	let mut names = HashSet::new();
	for _ in 0..count {
		let name = reader.name(MAX_NAME, "tensor name")?;
		if name.is_empty() || !names.insert(name.clone()) {
			return Err(corrupt("invalid or duplicate tensor name"));
		}
		let offset = reader.u64()?;
		let size = reader.u64()?;
		let dtype = ScalarType::parse(reader.u8()?)?;
		let rank = usize::from(reader.u8()?);
		if rank > MAX_RANK {
			return Err(corrupt("invalid tensor rank"));
		}
		let encoding = TensorEncoding::parse(reader.u8()?)?;
		let block_size = reader.u8()?;
		if reader.take(4)?.iter().any(|byte| *byte != 0) {
			return Err(corrupt("tensor reserved bytes are nonzero"));
		}
		let mut shape = Vec::with_capacity(rank);
		for dimension in 0..MAX_RANK {
			let extent = reader.u64()?;
			if dimension < rank {
				shape.push(extent);
			}
		}
		metadata.push((name, offset, size, dtype, shape, encoding, block_size));
	}
	let blob = &bytes[index_size..];
	let mut ranges = Vec::new();
	let mut tensors = Vec::with_capacity(count);
	for (name, offset, size, dtype, shape, encoding, block_size) in metadata {
		let start = usize::try_from(offset).map_err(|_| corrupt("tensor offset exceeds usize"))?;
		let size = usize::try_from(size).map_err(|_| corrupt("tensor size exceeds usize"))?;
		let end = start
			.checked_add(size)
			.ok_or_else(|| corrupt("tensor range overflow"))?;
		let data = blob
			.get(start..end)
			.ok_or_else(|| corrupt("tensor payload range is outside its section"))?
			.to_vec();
		let tensor = Tensor {
			name,
			dtype,
			shape,
			encoding,
			block_size,
			data,
		};
		tensor.validate(weight, version, true)?;
		if size != 0 {
			ranges.push((start, end));
		}
		tensors.push(tensor);
	}
	ranges.sort_unstable_by_key(|range| range.0);
	if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
		return Err(corrupt("overlapping tensor payload ranges"));
	}
	Ok(tensors)
}

fn encode_optimizer(optimizer: &Optimizer) -> Result<Vec<u8>> {
	if !matches!(optimizer.kind.as_str(), "SGD" | "Adam" | "AdamW" | "Muon") {
		return Err(Error::invalid_argument("unknown .oam optimizer type"));
	}
	if optimizer.kind == "Muon" {
		if !optimizer.second_moment.is_empty() {
			return Err(Error::invalid_argument(
				"Muon .oam state has a second moment",
			));
		}
	} else if optimizer.first_moment.len() != optimizer.second_moment.len() {
		return Err(Error::invalid_argument(
			".oam optimizer moment lengths differ",
		));
	}
	let mut bytes = Vec::with_capacity(
		OPTIMIZER_SIZE
			+ (optimizer.first_moment.len() + optimizer.second_moment.len()) * size_of::<f32>(),
	);
	push_name(&mut bytes, &optimizer.kind, 16, "optimizer type")?;
	for value in [
		optimizer.learning_rate,
		optimizer.beta1,
		optimizer.beta2,
		optimizer.epsilon,
		optimizer.weight_decay,
	] {
		push_f32(&mut bytes, value);
	}
	push_i64(&mut bytes, optimizer.step);
	push_u64(
		&mut bytes,
		u64::try_from(optimizer.first_moment.len())
			.map_err(|_| Error::resource_exhausted("optimizer state exceeds u64"))?,
	);
	bytes.extend_from_slice(&[0; 16]);
	for values in [&optimizer.first_moment, &optimizer.second_moment] {
		for value in values {
			push_f32(&mut bytes, *value);
		}
	}
	Ok(bytes)
}

fn decode_optimizer(bytes: &[u8]) -> Result<Optimizer> {
	if bytes.len() < OPTIMIZER_SIZE {
		return Err(corrupt("invalid optimizer header"));
	}
	let mut reader = Reader::new(bytes);
	let kind = reader.name(16, "optimizer type")?;
	if !matches!(kind.as_str(), "SGD" | "Adam" | "AdamW" | "Muon") {
		return Err(corrupt("unknown optimizer type"));
	}
	let learning_rate = reader.f32()?;
	let beta1 = reader.f32()?;
	let beta2 = reader.f32()?;
	let epsilon = reader.f32()?;
	let weight_decay = reader.f32()?;
	let step = reader.i64()?;
	let count =
		usize::try_from(reader.u64()?).map_err(|_| corrupt("optimizer count exceeds usize"))?;
	reader.skip(16)?;
	let arrays = if kind == "Muon" { 1 } else { 2 };
	let expected = count
		.checked_mul(arrays)
		.and_then(|count| count.checked_mul(size_of::<f32>()))
		.and_then(|size| size.checked_add(OPTIMIZER_SIZE))
		.ok_or_else(|| corrupt("optimizer size overflow"))?;
	if expected != bytes.len() {
		return Err(corrupt("optimizer payload size mismatch"));
	}
	let mut read_values = || -> Result<Vec<f32>> { (0..count).map(|_| reader.f32()).collect() };
	let first_moment = read_values()?;
	let second_moment = if arrays == 2 {
		read_values()?
	} else {
		Vec::new()
	};
	Ok(Optimizer {
		kind,
		learning_rate,
		beta1,
		beta2,
		epsilon,
		weight_decay,
		step,
		first_moment,
		second_moment,
	})
}

fn encode_progress(progress: &Progress) -> Result<Vec<u8>> {
	let mut bytes = Vec::with_capacity(PROGRESS_SIZE);
	bytes.push(progress.phase);
	bytes.extend_from_slice(&[0; 3]);
	push_i64(&mut bytes, progress.step);
	push_u64(&mut bytes, progress.bytes_seen);
	push_u64(&mut bytes, progress.env_steps);
	push_f32(&mut bytes, progress.learning_rate);
	push_f32(&mut bytes, progress.best_metric);
	bytes.push(u8::from(progress.lower_is_better));
	bytes.extend_from_slice(&[0; 3]);
	push_name(&mut bytes, &progress.metric_name, 32, "metric name")?;
	bytes.extend_from_slice(&[0; 32]);
	debug_assert_eq!(bytes.len(), PROGRESS_SIZE);
	Ok(bytes)
}

fn decode_progress(bytes: &[u8]) -> Result<Progress> {
	if bytes.len() != PROGRESS_SIZE {
		return Err(corrupt("invalid progress section"));
	}
	let mut reader = Reader::new(bytes);
	let phase = reader.u8()?;
	reader.skip(3)?;
	let step = reader.i64()?;
	let bytes_seen = reader.u64()?;
	let env_steps = reader.u64()?;
	let learning_rate = reader.f32()?;
	let best_metric = reader.f32()?;
	let lower_is_better = reader.u8()? != 0;
	reader.skip(3)?;
	let metric_name = reader.name(32, "metric name")?;
	reader.skip(32)?;
	Ok(Progress {
		phase,
		step,
		bytes_seen,
		env_steps,
		learning_rate,
		best_metric,
		lower_is_better,
		metric_name,
	})
}

fn validate_tensors(tensors: &[Tensor], weight: bool, version: u32, input: bool) -> Result<()> {
	let mut names = HashSet::new();
	for tensor in tensors {
		tensor.validate(weight, version, input)?;
		if !names.insert(&tensor.name) {
			return Err(if input {
				corrupt("duplicate tensor name")
			} else {
				Error::invalid_argument("duplicate .oam tensor name")
			});
		}
	}
	Ok(())
}

fn validate_name(name: &str, width: usize, label: &'static str) -> Result<()> {
	if name.as_bytes().contains(&0) || name.len() >= width {
		return Err(Error::invalid_argument(format!(
			".oam {label} must fit in {} non-NUL bytes",
			width - 1
		)));
	}
	Ok(())
}

fn push_name(bytes: &mut Vec<u8>, name: &str, width: usize, label: &'static str) -> Result<()> {
	validate_name(name, width, label)?;
	let start = bytes.len();
	bytes.extend_from_slice(name.as_bytes());
	bytes.resize(start + width, 0);
	Ok(())
}

fn page_align(value: usize) -> Result<usize> {
	value
		.checked_add(PAGE_SIZE - 1)
		.map(|value| value & !(PAGE_SIZE - 1))
		.ok_or_else(|| Error::resource_exhausted(".oam page alignment overflow"))
}

fn hash(bytes: &[u8]) -> u64 {
	bytes.iter().fold(FNV_OFFSET, |hash, byte| {
		(hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
	})
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
	if let Some(parent) = path
		.parent()
		.filter(|parent| !parent.as_os_str().is_empty())
	{
		fs::create_dir_all(parent).map_err(|source| Error::io("create .oam directory", source))?;
	}
	static SEQUENCE: AtomicU64 = AtomicU64::new(0);
	let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
	let mut temporary = path.as_os_str().to_owned();
	temporary.push(format!(".tmp.{}.{}", std::process::id(), sequence));
	let temporary = PathBuf::from(temporary);
	let result = (|| {
		let mut file = File::create(&temporary)
			.map_err(|source| Error::io("create temporary .oam", source))?;
		file.write_all(bytes)
			.map_err(|source| Error::io("write temporary .oam", source))?;
		file.sync_all()
			.map_err(|source| Error::io("sync temporary .oam", source))?;
		drop(file);
		fs::rename(&temporary, path).map_err(|source| Error::io("replace .oam", source))?;
		#[cfg(unix)]
		if let Some(parent) = path.parent() {
			File::open(if parent.as_os_str().is_empty() {
				Path::new(".")
			} else {
				parent
			})
			.and_then(|directory| directory.sync_all())
			.map_err(|source| Error::io("sync .oam directory", source))?;
		}
		Ok(())
	})();
	if result.is_err() {
		let _ = fs::remove_file(&temporary);
	}
	result
}

fn corrupt(message: impl Into<String>) -> Error {
	Error::checkpoint_corrupt(message)
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
	push_u32(bytes, value.to_bits());
}

struct Reader<'a> {
	bytes: &'a [u8],
	offset: usize,
}

impl<'a> Reader<'a> {
	const fn new(bytes: &'a [u8]) -> Self {
		Self { bytes, offset: 0 }
	}

	fn take(&mut self, count: usize) -> Result<&'a [u8]> {
		let end = self
			.offset
			.checked_add(count)
			.ok_or_else(|| corrupt("field offset overflow"))?;
		let value = self
			.bytes
			.get(self.offset..end)
			.ok_or_else(|| corrupt("truncated field"))?;
		self.offset = end;
		Ok(value)
	}

	fn skip(&mut self, count: usize) -> Result<()> {
		self.take(count).map(|_| ())
	}

	fn u8(&mut self) -> Result<u8> {
		Ok(self.take(1)?[0])
	}

	fn u32(&mut self) -> Result<u32> {
		Ok(u32::from_le_bytes(
			self.take(4)?.try_into().expect("four-byte field"),
		))
	}

	fn u64(&mut self) -> Result<u64> {
		Ok(u64::from_le_bytes(
			self.take(8)?.try_into().expect("eight-byte field"),
		))
	}

	fn i64(&mut self) -> Result<i64> {
		Ok(i64::from_le_bytes(
			self.take(8)?.try_into().expect("eight-byte field"),
		))
	}

	fn f32(&mut self) -> Result<f32> {
		Ok(f32::from_bits(self.u32()?))
	}

	fn name(&mut self, width: usize, label: &'static str) -> Result<String> {
		let bytes = self.take(width)?;
		let end = bytes
			.iter()
			.position(|byte| *byte == 0)
			.ok_or_else(|| corrupt(format!("{label} is not NUL terminated")))?;
		std::str::from_utf8(&bytes[..end])
			.map(str::to_owned)
			.map_err(|_| corrupt(format!("{label} is not UTF-8")))
	}
}
