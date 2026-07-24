#pragma once

#include <Oa/Core/MatrixShape.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaMatrix;

// Stable semantic metadata for an operation. This describes what an operation
// means before lowering chooses a kernel, launch geometry, queue, or device.
// Generated registries provide these descriptors from FnAutogen schemas.
enum class OaOperationValueKind : OaU8 {
	Matrix = 1,
	Image = 2,
	Audio = 3,
	VideoFrame = 4,
};

enum class OaOperationShapeRule : OaU8 {
	MatchInput,
	Broadcast,
	MatMulNt,
	// Output shapes are validated and allocated by the operation's private
	// lowering because they cannot be represented by a uniform binary rule.
	Explicit,
};

enum class OaOperationDtypeRule : OaU8 {
	MatchInput,
	PromoteFloat,
};

enum class OaOperationDifferentiation : OaU8 {
	None,
	Reverse,
};

enum class OaOperationLowering : OaU8 {
	Dispatch,
	Gemm,
};

enum class OaOperationControlFlow : OaU8 {
	StraightLine,
	Conditional,
	Loop,
};

enum class OaOperationEffect : OaU8 {
	None = 0,
	ReadInputs = 1U << 0U,
	WriteOutputs = 1U << 1U,
};

// Ordered non-value inputs that affect operation meaning. Matrices, images,
// audio values, and video frames remain semantic values; these attributes
// preserve scalar/configuration data independently of runtime push layouts.
enum class OaOperationAttributeKind : OaU8 {
	Boolean = 1,
	SignedInteger = 2,
	UnsignedInteger = 3,
	Float = 4,
	String = 5,
	Shape = 6,
	Enum = 7,
};

class OaOperationAttribute {
public:
	OaString Name;
	OaOperationAttributeKind Kind = OaOperationAttributeKind::Boolean;
	OaBool Boolean = false;
	OaI64 SignedInteger = 0;
	OaU64 UnsignedInteger = 0;
	OaF64 Float = 0.0;
	OaString Text;
	OaMatrixShape Shape{};

	[[nodiscard]] static OaOperationAttribute FromBoolean(
		OaStringView InName, OaBool InValue);
	[[nodiscard]] static OaOperationAttribute FromSignedInteger(
		OaStringView InName, OaI64 InValue);
	[[nodiscard]] static OaOperationAttribute FromUnsignedInteger(
		OaStringView InName, OaU64 InValue);
	[[nodiscard]] static OaOperationAttribute FromFloat(
		OaStringView InName, OaF64 InValue);
	[[nodiscard]] static OaOperationAttribute FromString(
		OaStringView InName, OaString InValue);
	[[nodiscard]] static OaOperationAttribute FromShape(
		OaStringView InName, const OaMatrixShape& InValue);
	[[nodiscard]] static OaOperationAttribute FromEnum(
		OaStringView InName, OaString InSymbol);
	[[nodiscard]] OaStatus Validate() const;
};

[[nodiscard]] constexpr OaOperationEffect operator|(
	OaOperationEffect InA, OaOperationEffect InB) noexcept
{
	return static_cast<OaOperationEffect>(
		static_cast<OaU8>(InA) | static_cast<OaU8>(InB));
}

class OaOperationContract {
public:
	static constexpr OaU8 MaxValues = 8U;
	static constexpr OaU8 MaxAttributes = 8U;
	static constexpr OaU8 NoAliasInput = 0x0fU;

	OaStringView Name;
	OaU64 Hash = 0;
	OaU32 InputKinds = 0;
	OaU32 OutputKinds = 0;
	OaU8 InputCount = 0;
	OaU8 OutputCount = 0;
	OaU8 AttributeCount = 0;
	// FNV-1a over each ordered attribute kind, UTF-8 name, and a zero
	// terminator. Zero is the canonical signature for an empty list.
	OaU64 AttributeSignatureHash = 0;
	OaOperationShapeRule ShapeRule = OaOperationShapeRule::MatchInput;
	OaOperationDtypeRule DtypeRule = OaOperationDtypeRule::MatchInput;
	OaOperationDifferentiation Differentiation = OaOperationDifferentiation::None;
	OaOperationLowering Lowering = OaOperationLowering::Dispatch;
	OaOperationEffect Effects = OaOperationEffect::None;
	// One bit per logical input. Mutation describes semantic state change, not
	// merely reuse of a writable allocation during lowering.
	OaU8 MutatedInputMask = 0U;
	// One four-bit input index per output; 0xf means the output does not alias
	// an input. This compact form matches the eight-value schema limit.
	OaU32 OutputAliasInputs = UINT32_MAX;
	OaOperationControlFlow ControlFlow = OaOperationControlFlow::StraightLine;

	[[nodiscard]] constexpr OaBool MutatesInput(OaU32 InIndex) const noexcept {
		return InIndex < MaxValues
			and (MutatedInputMask & static_cast<OaU8>(1U << InIndex)) != 0U;
	}
	[[nodiscard]] constexpr OaU8 AliasInputForOutput(
		OaU32 InIndex) const noexcept
	{
		return InIndex < MaxValues
			? static_cast<OaU8>((OutputAliasInputs >> (InIndex * 4U)) & 0x0fU)
			: NoAliasInput;
	}
};

[[nodiscard]] OaU64 OaOperationAttributeSignatureHash(
	OaSpan<const OaOperationAttribute> InAttributes) noexcept;
[[nodiscard]] OaStatus OaValidateOperationAttributes(
	const OaOperationContract& InContract,
	OaSpan<const OaOperationAttribute> InAttributes);

// Schema-driven semantic validation and shape inference. Lowerings call these
// before allocating outputs; kernel choice and launch geometry remain separate.
[[nodiscard]] OaStatus OaValidateBinaryOperation(
	const OaOperationContract& InContract,
	const OaMatrix& InA,
	const OaMatrix& InB
);
[[nodiscard]] OaResult<OaMatrixShape> OaInferBinaryOperationShape(
	const OaOperationContract& InContract,
	const OaMatrix& InA,
	const OaMatrix& InB
);

// Generated semantic contracts are publicly inspectable, while their source
// of truth remains the FnAutogen schemas.
#include <Oa/Core/OperationRegistry.gen.h>
