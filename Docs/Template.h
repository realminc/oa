// Oa header template — copy and adapt for new types.
//
// NAMING CONVENTION — Infrastructure vs Implementations
// ======================================================
// Prefix "OaDg" marks infrastructure classes (dependency graph core):
//   OaDgNode        -- base node (users derive OaNodeWeight, OaNodeInput, etc)
//   OaDgGraph       -- DAG evaluator container
//   OaDgPlug        -- typed connection point (data flow, dirty state)
//   OaDgHandle      -- lightweight stable reference for editors/serialization
//
// Concrete implementations drop the Dg prefix (they're built on OaDg):
//   OaNodeWeight    -- trainable parameter (extends OaDgNode)
//   OaAddNode       -- concrete add operation (extends OaDgNode)
//   OaVkNode        -- GPU dispatch base (extends OaDgNode)
//
// This makes intent clear: "OaDg*" = infrastructure layer, "Oa*" = application.
//
// STYLE (C with classes, one keyword):
//   Always `class`. Never `struct` for user-defined types — same machinery, but `struct`
//   defaults to public in a way that fights a single layout. We start with `public:` and
//   add `private:` only when something must be hidden.
//
// LAYOUT:
//   1. Related includes: Oa (grouped by area) → system STL → math / third-party
//   2. File-scope constants
//   3. Types: class body = public first (data, ctors, methods, ops), then private if needed
//
// MEMBERS & METHODS:
//   - Members: trailing underscore (mName, mShape)
//   - Methods: PascalCase, Get/Set prefixes for accessors
//   - [[nodiscard]] on methods where ignoring result would be a bug (queries, status checks)
//   - noexcept on methods that won't throw (simple ops, getters, setters)
//
// CLASS BODY ORDER (public section):
//   1. Data members (public, for Python binding exposure)
//   2. Constructors
//   3. Destructors
//   4. Methods (primary API)
//   5. Operators (syntactic sugar, secondary)
//   6. Internal API section (if needed, with data members)
//   7. Private section (mirrors public order if needed)

#pragma once
// Oa includes first, alphabetical module order.

// Oa.
#include <Oa.h>
// Oa Core.
#include <Oa/Core.h>
#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Std/Allocator.h>
#include <Oa/Core/Std/StringView.h>
#include <Oa/Core/Types.h>
// Oa Data.
// Oa Crypto.
// Oa Ml.
#include <Oa/Ml/FnMatrix.h>
// Oa Network.
// Oa Runtime.
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Graph.h>
// Oa Ui.
// Oa Vision.

// System includes, std and friends.
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

// Math / external.
#include <cmath>

// Third Party.


class OaClassTemplate {
public:
	// Type with hidden detail: public API first, private last.

	// Data, class members (public for Python binding exposure).
	OaVec<OaI64> InShape;

	// Constructors.
	OaClassTemplate() = default;
	explicit OaClassTemplate(const OaVec<OaI64>& InShape) noexcept
		: InShape(InShape)
		, Enabled_(false)
	{}

	// Destructors.
	~OaClassTemplate() = default;

	// Methods (primary API).
	[[nodiscard]] bool IsValid() const noexcept;
	void Backward();
	void ZeroGrad() noexcept;
	void RetainGrad() noexcept;
	void AccumulateGrad(const OaDeviceMatrix& InGrad);

	[[nodiscard]] bool Enabled() const noexcept { return Enabled_; }
	[[nodiscard]] const OaVec<OaI64>& Shape() const noexcept { return InShape; }

	void SetEnabled(bool InValue) noexcept { Enabled_ = InValue; }
	void SetShape(const OaVec<OaI64>& InShape) noexcept { Shape_ = InShape; }

	// Operators (syntactic sugar, secondary).
	OaClassTemplate(const OaClassTemplate&) = default;
	OaClassTemplate& operator=(const OaClassTemplate&) = default;

	OaClassTemplate(OaClassTemplate&&) noexcept = default;
	OaClassTemplate& operator=(OaClassTemplate&&) noexcept = default;

	// Internal API - used by OaFnClassTemplate namespace functions.
public:
	void SyncInternal() const noexcept;

	// Internal data members.
	OaVec<OaI64> Shape_;
	bool Enabled_;

private:
	// Same order / structure as public if needed.

	void UpdateInternal() noexcept;
};


// ============================================================================
// OaFn* NAMESPACE TEMPLATE
// ============================================================================
// Purpose: OaFn* namespaces provide C-style function APIs that mirror Oa* class methods.
// This dual syntax allows both OOP (object.method()) and functional (Fn::method(obj))
// usage patterns. The class is typically all-public, and the Fn namespace wraps
// it with class-like syntax using the first parameter as the receiver.
//
// NAMING CONVENTION:
//   - OaClass: the data class (e.g., OaMatrix, OaImage)
//   - OaFnClass: the namespace of functions (e.g., OaFnMatrix, OaFnImage)
//   - Functions follow PascalCase, matching class method names
//
// FUNCTION ORDERING:
//   1. Configuration functions (setters/getters for global state)
//   2. Factory functions (constructors: Empty, Zeros, Ones, Full, Rand, etc.)
//   3. Transfer functions (host ↔ device)
//   4. Core operations (math, transforms)
//   5. Shape/indexing helpers
//   6. Domain-specific operations
//
// STYLE:
//   - [[nodiscard]] on functions returning values
//   - noexcept on functions that won't throw
//   - const OaClass& for read-only operations
//   - OaClass& for in-place mutations
//   - In* prefix for input parameters (InShape, InValue, InOther)


// Helper struct example (optional).
struct OaFnResult {
	OaClassTemplate Value;
	OaI32 Index;
};


// OaFnClassTemplate — namespace-based API for ClassTemplate operations.
// Usage: OaFnClassTemplate::Zeros({10, 20}), OaFnClassTemplate::Process(obj), etc.
namespace OaFnClassTemplate {

// --- Configuration ---
void SetConfig(OaI32 InValue);
[[nodiscard]] OaI32 GetConfig();

// --- Factory functions ---
[[nodiscard]] OaClassTemplate Empty(OaMatrixShape InShape);
[[nodiscard]] OaClassTemplate Zeros(OaMatrixShape InShape);
[[nodiscard]] OaClassTemplate Ones(OaMatrixShape InShape);
[[nodiscard]] OaClassTemplate Full(OaMatrixShape InShape, OaF64 InValue);
[[nodiscard]] OaClassTemplate Rand(OaMatrixShape InShape);
[[nodiscard]] OaClassTemplate FromBytes(OaSpan<const OaU8> InData, OaMatrixShape InShape);

// --- Transfer functions ---
[[nodiscard]] OaStatus CopyToHost(const OaClassTemplate& InSrc, void* OutHost, OaU64 InBytes);
[[nodiscard]] OaF32 Scalar(const OaClassTemplate& InSrc);

// --- Core operations ---
void AddInPlace(OaClassTemplate& InSelf, const OaClassTemplate& InOther);
void ScaleInPlace(OaClassTemplate& InSelf, OaF32 InScalar);
void Fill(OaClassTemplate& InSelf, OaF32 InValue);

[[nodiscard]] OaClassTemplate Add(const OaClassTemplate& InA, const OaClassTemplate& InB);
[[nodiscard]] OaClassTemplate Sub(const OaClassTemplate& InA, const OaClassTemplate& InB);
[[nodiscard]] OaClassTemplate Mul(const OaClassTemplate& InA, const OaClassTemplate& InB);
[[nodiscard]] OaClassTemplate Div(const OaClassTemplate& InA, const OaClassTemplate& InB);

[[nodiscard]] OaClassTemplate Sum(const OaClassTemplate& InA, OaI32 InDim = -1);
[[nodiscard]] OaClassTemplate Mean(const OaClassTemplate& InA, OaI32 InDim = -1);
[[nodiscard]] OaClassTemplate Max(const OaClassTemplate& InA, OaI32 InDim = -1);

// --- Shape & indexing helpers ---
[[nodiscard]] OaClassTemplate Reshape(const OaClassTemplate& InA, OaMatrixShape InShape);
[[nodiscard]] OaClassTemplate Transpose(const OaClassTemplate& InA, OaI32 InDim0, OaI32 InDim1);
[[nodiscard]] OaClassTemplate View(const OaClassTemplate& InA, OaMatrixShape InNewShape);
[[nodiscard]] OaClassTemplate Flatten(const OaClassTemplate& InA);

// --- Domain-specific operations ---
[[nodiscard]] OaFnResult Process(const OaClassTemplate& InA);
[[nodiscard]] OaClassTemplate Transform(const OaClassTemplate& InA, OaF32 InParam);

} // namespace OaFnClassTemplate


// TEMPLATE.CPP — IMPLEMENTATION HOOKUP PATTERN.
//
// This section shows how class methods/operators delegate to OaFn* namespace
// functions. The actual implementation lives in the Fn namespace; the class
// provides syntactic sugar (operators, constructors) that forward to it.
//
// KEY PATTERNS:
//   1. Class operators → OaFn:: functions (e.g., operator+ calls OaFn::Add)
//   2. Class constructors → OaFn:: factory functions (e.g., ctor calls OaFn::Full)
//   3. OaFn:: functions contain the actual implementation logic
//   4. Both class and Fn can call the same underlying implementation
//
// See Source/Private/Oa/Core/Matrix/Matrix.cpp for real examples.

/*

// Template.cpp — Example implementation file.

#include <Oa/Core/Template.h>
#include <Oa/Core/FnTemplate.h>

// OaClassTemplate member functions.

// Constructor delegates to OaFnClassTemplate::Full
// This enables braced-init syntax: OaClassTemplate obj = {shape, value};
OaClassTemplate::OaClassTemplate(OaMatrixShape InShape, OaF32 InFillValue) {
	*this = OaFnClassTemplate::Full(InShape, static_cast<OaF64>(InFillValue));
}

// OaClassTemplate operator overloads.

// Arithmetic operators delegate to OaFnClassTemplate functions
OaClassTemplate OaClassTemplate::operator+(const OaClassTemplate& InOther) const {
	return OaFnClassTemplate::Add(*this, InOther);
}

OaClassTemplate OaClassTemplate::operator-(const OaClassTemplate& InOther) const {
	return OaFnClassTemplate::Sub(*this, InOther);
}

OaClassTemplate OaClassTemplate::operator*(const OaClassTemplate& InOther) const {
	return OaFnClassTemplate::Mul(*this, InOther);
}

OaClassTemplate OaClassTemplate::operator/(const OaClassTemplate& InOther) const {
	return OaFnClassTemplate::Div(*this, InOther);
}

// Scalar operators
OaClassTemplate OaClassTemplate::operator+(OaF32 InScalar) const {
	return OaFnClassTemplate::Add(*this, OaFnClassTemplate::Full(Shape_, static_cast<OaF64>(InScalar)));
}

OaClassTemplate OaClassTemplate::operator*(OaF32 InScalar) const {
	return OaFnClassTemplate::Scale(*this, InScalar);
}

// Unary operators
OaClassTemplate OaClassTemplate::operator-() const {
	return OaFnClassTemplate::Neg(*this);
}

// Compound assignment operators
OaClassTemplate& OaClassTemplate::operator+=(const OaClassTemplate& InOther) {
	OaFnClassTemplate::AddInPlace(*this, InOther);
	return *this;
}

OaClassTemplate& OaClassTemplate::operator*=(OaF32 InScalar) {
	OaFnClassTemplate::ScaleInPlace(*this, InScalar);
	return *this;
}

// OaFnClassTemplate implementation.

// Configuration functions
static OaI32 GConfig = 0;

void OaFnClassTemplate::SetConfig(OaI32 InValue) {
	GConfig = InValue;
}

OaI32 OaFnClassTemplate::GetConfig() {
	return GConfig;
}

// Factory functions (actual implementation lives here)
OaClassTemplate OaFnClassTemplate::Empty(OaMatrixShape InShape) {
	OaClassTemplate result;
	result.InShape = InShape;
	result.Shape_ = InShape;
	return result;
}

OaClassTemplate OaFnClassTemplate::Full(OaMatrixShape InShape, OaF64 InValue) {
	OaClassTemplate result = Empty(InShape);
	// ... actual fill implementation ...
	return result;
}

// Core operations (actual implementation lives here)
void OaFnClassTemplate::AddInPlace(OaClassTemplate& InSelf, const OaClassTemplate& InOther) {
	// ... actual add implementation ...
}

OaClassTemplate OaFnClassTemplate::Add(const OaClassTemplate& InA, const OaClassTemplate& InB) {
	// ... actual add implementation ...
}

*/
