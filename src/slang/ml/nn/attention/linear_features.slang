// Stable Katharopoulos feature map; negative values never subtract/add one.
float phi(float x) { return x >= 0.0f ? x + 1.0f : exp(x); }
float phi_grad(float x) { return x >= 0.0f ? 1.0f : exp(x); }
