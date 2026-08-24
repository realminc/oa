#include <oa/core/color.h>
#include <oa/core/fnColor.h>

// oa::Color operator implementations wrap oa::FnColor namespace functions.
// The pattern matches oa::Matrix/oa::FnMatrix.

// ─── Methods ──────────────────────────────────────────────────────────────────

oa::Color oa::Color::lerp(const oa::Color& inOther, oa::F32 inT) const {
	return oa::FnColor::lerp(*this, inOther, inT);
}

// ─── Arithmetic operators ─────────────────────────────────────────────────────

oa::Color oa::Color::operator+(const oa::Color& inOther) const {
	return oa::FnColor::add(*this, inOther);
}

oa::Color oa::Color::operator-(const oa::Color& inOther) const {
	return oa::FnColor::sub(*this, inOther);
}

oa::Color oa::Color::operator*(const oa::Color& inOther) const {
	return oa::FnColor::mul(*this, inOther);
}

oa::Color oa::Color::operator*(oa::F32 inScalar) const {
	return oa::FnColor::scale(*this, inScalar);
}

oa::Color oa::Color::operator/(oa::F32 inScalar) const {
	return oa::FnColor::div(*this, inScalar);
}

// ─── Compound assignment operators ────────────────────────────────────────────

oa::Color& oa::Color::operator+=(const oa::Color& inOther) {
	*this = oa::FnColor::add(*this, inOther);
	return *this;
}

oa::Color& oa::Color::operator-=(const oa::Color& inOther) {
	*this = oa::FnColor::sub(*this, inOther);
	return *this;
}

oa::Color& oa::Color::operator*=(const oa::Color& inOther) {
	*this = oa::FnColor::mul(*this, inOther);
	return *this;
}

oa::Color& oa::Color::operator*=(oa::F32 inScalar) {
	*this = oa::FnColor::scale(*this, inScalar);
	return *this;
}

oa::Color& oa::Color::operator/=(oa::F32 inScalar) {
	*this = oa::FnColor::div(*this, inScalar);
	return *this;
}
