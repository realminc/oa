// oa::complex — Complex arithmetic for Slang shaders
//
// Import: `import complex;` (slangc -I to Core/Math or installed lib/Oa/Core/shaders/Math)
// Used by SSM complex scan kernels (cTEM).
// Pure math — no bindings, no push constants, no entry points.

module complex;

public struct Complex {
	public float r;
	public float i;
};

// Complex multiply: (ar + ai*i) * (br + bi*i)
public Complex complexMul(float ar, float ai, float br, float bi) {
	Complex result;
	result.r = ar * br - ai * bi;
	result.i = ar * bi + ai * br;
	return result;
}

// Complex multiply with conjugate of second: (ar + ai*i) * (br - bi*i)
public Complex complexMulConjugate(float ar, float ai, float br, float bi) {
	Complex result;
	result.r = ar * br + ai * bi;
	result.i = -ar * bi + ai * br;
	return result;
}

// Complex add
public Complex complexAdd(float ar, float ai, float br, float bi) {
	Complex result;
	result.r = ar + br;
	result.i = ai + bi;
	return result;
}
