// OA Slang attributes for kernel metadata

// Operation family attribute
[__AttributeUsage(_AttributeTargets.Function)]
struct kernel_name {
	string op;
}

// Domain attribute
[__AttributeUsage(_AttributeTargets.Function)]
struct domain {
	string domain;
}

// Variant attribute for kernel variants
[__AttributeUsage(_AttributeTargets.Function)]
struct variant {
	string name;
}

// Capability requirements
[__AttributeUsage(_AttributeTargets.Function)]
struct requires {
	string capability;
}

// Data type support
[__AttributeUsage(_AttributeTargets.Function)]
struct dtype {
	string dtype;
}

// Layout requirements
[__AttributeUsage(_AttributeTargets.Function)]
struct layout {
	string layout;
}

// Experimental/stable status
[__AttributeUsage(_AttributeTargets.Function)]
struct status {
	string status; // "experimental", "stable", "deprecated"
}

// Autotune family
[__AttributeUsage(_AttributeTargets.Function)]
struct tune_family {
	string family;
}
