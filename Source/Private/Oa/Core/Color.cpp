#include <Oa/Core/Color.h>
#include <Oa/Core/FnColor.h>

// OaColor operator implementations — wrap OaFnColor namespace functions.
// Pattern matches OaMatrix/OaFnMatrix (see Matrix.cpp).

// ─── Methods ──────────────────────────────────────────────────────────────────

OaColor OaColor::Lerp(const OaColor& InOther, OaF32 InT) const {
	return OaFnColor::Lerp(*this, InOther, InT);
}

// ─── Arithmetic operators ─────────────────────────────────────────────────────

OaColor OaColor::operator+(const OaColor& InOther) const {
	return OaFnColor::Add(*this, InOther);
}

OaColor OaColor::operator-(const OaColor& InOther) const {
	return OaFnColor::Sub(*this, InOther);
}

OaColor OaColor::operator*(const OaColor& InOther) const {
	return OaFnColor::Mul(*this, InOther);
}

OaColor OaColor::operator*(OaF32 InScalar) const {
	return OaFnColor::Scale(*this, InScalar);
}

OaColor OaColor::operator/(OaF32 InScalar) const {
	return OaFnColor::Div(*this, InScalar);
}

// ─── Compound assignment operators ────────────────────────────────────────────

OaColor& OaColor::operator+=(const OaColor& InOther) {
	*this = OaFnColor::Add(*this, InOther);
	return *this;
}

OaColor& OaColor::operator-=(const OaColor& InOther) {
	*this = OaFnColor::Sub(*this, InOther);
	return *this;
}

OaColor& OaColor::operator*=(const OaColor& InOther) {
	*this = OaFnColor::Mul(*this, InOther);
	return *this;
}

OaColor& OaColor::operator*=(OaF32 InScalar) {
	*this = OaFnColor::Scale(*this, InScalar);
	return *this;
}

OaColor& OaColor::operator/=(OaF32 InScalar) {
	*this = OaFnColor::Div(*this, InScalar);
	return *this;
}

