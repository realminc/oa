// Matrix Dropout mapping built on the Core RNG Philox primitive.
module dropout_rng;

import philox;

public float apply_dropout(float value, uint random_bits, float probability) {
	float keep = philox_to_float01(random_bits) >= probability ? 1.0f : 0.0f;
	return value * keep * (1.0f / (1.0f - probability));
}
