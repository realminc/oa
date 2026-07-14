// oa::complex — Complex arithmetic for Slang shaders
//
// Import: `import Complex;` (slangc -I to Core/Math or installed lib/Oa/Core/shaders/Math)
// Used by SSM complex scan kernels (cTEM).
// Pure math — no bindings, no push constants, no entry points.

module Complex;

public struct OaComplex {
	public float r;
	public float i;
};

// Complex multiply: (ar + ai*i) * (br + bi*i)
public OaComplex Cmul(float ar, float ai, float br, float bi) {
	OaComplex result;
	result.r = ar * br - ai * bi;
	result.i = ar * bi + ai * br;
	return result;
}

// Complex multiply with conjugate of second: (ar + ai*i) * (br - bi*i)
public OaComplex CmulConj(float ar, float ai, float br, float bi) {
	OaComplex result;
	result.r = ar * br + ai * bi;
	result.i = -ar * bi + ai * br;
	return result;
}

// Complex add
public OaComplex Cadd(float ar, float ai, float br, float bi) {
	OaComplex result;
	result.r = ar + br;
	result.i = ai + bi;
	return result;
}
