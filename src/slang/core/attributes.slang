// Core OA Slang attributes for kernel metadata.

module attributes;

// Operation family attribute
[__AttributeUsage(_AttributeTargets.Function)]
public struct kernel_name {
	public string op;
}

// Domain attribute
[__AttributeUsage(_AttributeTargets.Function)]
public struct domain {
	public string domain;
}

// Variant attribute for kernel variants
[__AttributeUsage(_AttributeTargets.Function)]
public struct variant {
	public string name;
}

// Capability requirements
[__AttributeUsage(_AttributeTargets.Function)]
public struct requires {
	public string capability;
}

// Dense storage dtype. Values use normalized lowercase scalar tokens matching
// Rust primitive spellings where available: f16, bf16, f32, f64, i8, u8, ...
// This does not describe accumulator precision or packed quantization policy.
[__AttributeUsage(_AttributeTargets.Function)]
public struct dtype {
	public string dtype;
}

// Layout requirements
[__AttributeUsage(_AttributeTargets.Function)]
public struct layout {
	public string layout;
}

// Experimental/stable status
[__AttributeUsage(_AttributeTargets.Function)]
public struct status {
	public string status; // "experimental", "stable", "deprecated"
}

// Autotune family
[__AttributeUsage(_AttributeTargets.Function)]
public struct tune_family {
	public string family;
}
