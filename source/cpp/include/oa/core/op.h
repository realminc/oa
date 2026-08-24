#pragma once

#include <oa/core/matrixShape.h>
#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa { class Matrix; }

// Stable semantic metadata for an operation. This describes what an operation
// means before lowering chooses a kernel, launch geometry, queue, or device.
// generated registries provide these descriptors from FnAutogen schemas.

namespace oa {

enum class OpValueKind : oa::U8 {
	None = 0,
	Matrix = 1,
	Image = 2,
	Audio = 3,
	VideoFrame = 4,
	QuantMatrix = 5,
};

enum class OpShapeRule : oa::U8 {
	MatchInput,
	Broadcast,
	MatMulNt,
	// output shapes are validated and allocated by the operation's private
	// lowering because they cannot be represented by a uniform binary rule.
	Explicit,
};

enum class OpDtypeRule : oa::U8 {
	MatchInput,
	PromoteFloat,
	// input and output dtypes are validated by operation-specific lowering.
	// This covers heterogeneous contracts such as labels -> metric counters.
	Explicit,
};

enum class OpDifferentiation : oa::U8 {
	None,
	Reverse,
};

enum class OpLowering : oa::U8 {
	Dispatch,
	Gemm,
};

enum class OpControlFlow : oa::U8 {
	StraightLine,
	Conditional,
	Loop,
};

enum class OpEffect : oa::U8 {
	None = 0,
	ReadInputs = 1U << 0U,
	WriteOutputs = 1U << 1U,
};

// Ordered non-value inputs that affect operation meaning. matrices, images,
// audio values, and video frames remain semantic values; these attributes
// preserve scalar/configuration data independently of runtime push layouts.
enum class OpAttributeKind : oa::U8 {
	Boolean = 1,
	SignedInteger = 2,
	UnsignedInteger = 3,
	Float = 4,
	String = 5,
	Shape = 6,
	Enum = 7,
};

class OpAttribute {
public:
	oa::String name;
	OpAttributeKind kind = OpAttributeKind::Boolean;
	oa::Bool boolean = false;
	oa::I64 signedInteger = 0;
	oa::U64 unsignedInteger = 0;
	oa::F64 floatVal = 0.0;
	oa::String text;
	MatrixShape shape{};

	[[nodiscard]] static OpAttribute fromBoolean(
		oa::StringView inName, oa::Bool inValue);
	[[nodiscard]] static OpAttribute fromSignedInteger(
		oa::StringView inName, oa::I64 inValue);
	[[nodiscard]] static OpAttribute fromUnsignedInteger(
		oa::StringView inName, oa::U64 inValue);
	[[nodiscard]] static OpAttribute fromFloat(
		oa::StringView inName, oa::F64 inValue);
	[[nodiscard]] static OpAttribute fromString(
		oa::StringView inName, oa::String inValue);
	[[nodiscard]] static OpAttribute fromShape(
		oa::StringView inName, const MatrixShape& inValue);
	[[nodiscard]] static OpAttribute fromEnum(
		oa::StringView inName, oa::String inSymbol);
	[[nodiscard]] oa::Status validate() const;
};

[[nodiscard]] constexpr OpEffect operator|(
	OpEffect inA, OpEffect inB) noexcept
{
	return static_cast<OpEffect>(
		static_cast<oa::U8>(inA) | static_cast<oa::U8>(inB));
}

class OpContract {
public:
	// Fixed values retain compact masks, aliases, and packed kind descriptors.
	// A homogeneous variadic tail may extend either side beyond this limit.
	static constexpr oa::U8 MaxValues = 8U;
	static constexpr oa::U8 MaxAttributes = 8U;
	static constexpr oa::U8 NoAliasInput = 0x0fU;

	oa::StringView name;
	oa::U64 hash = 0;
	oa::U32 inputKinds = 0;
	oa::U32 outputKinds = 0;
	oa::U8 inputCount = 0;
	oa::U8 outputCount = 0;
	OpValueKind variadicInputKind = OpValueKind::None;
	OpValueKind variadicOutputKind = OpValueKind::None;
	oa::U8 minimumVariadicInputCount = 0;
	oa::U8 minimumVariadicOutputCount = 0;
	oa::U8 attributeCount = 0;
	// FNV-1a over each ordered attribute kind, UTF-8 name, and a zero
	// terminator. Zero is the canonical signature for an empty list.
	oa::U64 attributeSignatureHash = 0;
	OpShapeRule shapeRule = OpShapeRule::MatchInput;
	OpDtypeRule dtypeRule = OpDtypeRule::MatchInput;
	OpDifferentiation differentiation = OpDifferentiation::None;
	OpLowering lowering = OpLowering::Dispatch;
	OpEffect effects = OpEffect::None;
	// One bit per logical input. Mutation describes semantic state change, not
	// merely reuse of a writable allocation during lowering.
	oa::U8 mutatedInputMask = 0U;
	// Optional logical inputs retain their contract slot and value kind. An
	// An absent input uses the semantic IR invalid-value identifier,
	// so overload/default selection does not erase present dependencies.
	oa::U8 optionalInputMask = 0U;
	// One four-bit input index per output; 0xf means the output does not alias
	// an input. This compact form matches the eight-value schema limit.
	oa::U32 outputAliasInputs = UINT32_MAX;
	OpControlFlow controlFlow = OpControlFlow::StraightLine;

	[[nodiscard]] constexpr oa::Bool hasVariadicInputs() const noexcept {
		return variadicInputKind != OpValueKind::None;
	}
	[[nodiscard]] constexpr oa::Bool hasVariadicOutputs() const noexcept {
		return variadicOutputKind != OpValueKind::None;
	}
	[[nodiscard]] constexpr oa::Bool acceptsInputCount(
		oa::Usize inCount) const noexcept
	{
		return hasVariadicInputs()
			? inCount >= static_cast<oa::Usize>(
				inputCount + minimumVariadicInputCount)
			: inCount == inputCount;
	}
	[[nodiscard]] constexpr oa::Bool acceptsOutputCount(
		oa::Usize inCount) const noexcept
	{
		return hasVariadicOutputs()
			? inCount >= static_cast<oa::Usize>(
				outputCount + minimumVariadicOutputCount)
			: inCount == outputCount;
	}
	[[nodiscard]] constexpr OpValueKind inputKindAt(
		oa::U32 inIndex) const noexcept
	{
		return inIndex < inputCount
			? static_cast<OpValueKind>(
				(inputKinds >> (inIndex * 4U)) & 0x0fU)
			: variadicInputKind;
	}
	[[nodiscard]] constexpr OpValueKind outputKindAt(
		oa::U32 inIndex) const noexcept
	{
		return inIndex < outputCount
			? static_cast<OpValueKind>(
				(outputKinds >> (inIndex * 4U)) & 0x0fU)
			: variadicOutputKind;
	}
	[[nodiscard]] constexpr oa::Bool mutatesInput(oa::U32 inIndex) const noexcept {
		return inIndex < MaxValues
			and (mutatedInputMask & static_cast<oa::U8>(1U << inIndex)) != 0U;
	}
	[[nodiscard]] constexpr oa::Bool isInputOptional(
		oa::U32 inIndex) const noexcept
	{
		return inIndex < MaxValues
			and (optionalInputMask & static_cast<oa::U8>(1U << inIndex)) != 0U;
	}
	[[nodiscard]] constexpr oa::U8 aliasInputForOutput(
		oa::U32 inIndex) const noexcept
	{
		return inIndex < MaxValues
			? static_cast<oa::U8>((outputAliasInputs >> (inIndex * 4U)) & 0x0fU)
			: NoAliasInput;
	}
};

[[nodiscard]] oa::U64 opAttributeSignatureHash(
	oa::Span<const OpAttribute> inAttributes) noexcept;
[[nodiscard]] oa::Status validateOpAttributes(
	const OpContract& inContract,
	oa::Span<const OpAttribute> inAttributes);

// Schema-driven semantic validation and shape inference. Lowerings call these
// before allocating outputs; kernel choice and launch geometry remain separate.
[[nodiscard]] oa::Status validateBinaryOp(
	const OpContract& inContract,
	const Matrix& inA,
	const Matrix& inB
);
[[nodiscard]] oa::Result<MatrixShape> inferBinaryOpShape(
	const OpContract& inContract,
	const Matrix& inA,
	const Matrix& inB
);

} // namespace oa

// generated semantic contracts are publicly inspectable, while their source
// of truth remains the FnAutogen schemas.
#include <oa/core/opRegistry.gen.h>
