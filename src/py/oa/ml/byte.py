"""Raw-byte model input and logit decoding."""

from .._native import (
	ml_byte_decode as decode,
	ml_byte_decode_text as decode_text,
	ml_byte_encode as encode,
	ml_byte_encode_batched as encode_batched,
	ml_byte_encode_text as encode_text,
	ml_byte_sample as sample,
)

BOS = 0x01
EOS = 0x02
PAD = 0x00
SEP = 0x03
VOCAB_SIZE = 256

__all__ = [
	"BOS",
	"EOS",
	"PAD",
	"SEP",
	"VOCAB_SIZE",
	"decode",
	"decode_text",
	"encode",
	"encode_batched",
	"encode_text",
	"sample",
]
