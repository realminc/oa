//! Shader metadata structures

/// Kernel descriptor
#[derive(Debug, Clone)]
pub struct KernelDesc {
    pub module: &'static str,
    pub entry: &'static str,
    pub domain: Domain,
    pub op: Op,
    pub variant: &'static str,
    pub requirements: CapabilitySet,
    pub dtypes: &'static [DType],
}

/// Operation domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Matrix,
    Vision,
    Ml,
    Audio,
    Video,
    Render,
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    MatMul,
    Softmax,
    Resize,
    // TODO: Add more operations
}

/// Capability requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySet {
    pub flags: CapabilityFlags,
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self {
            flags: CapabilityFlags::NONE,
        }
    }
}

bitflags::bitflags! {
    /// GPU capability flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CapabilityFlags: u64 {
        const NONE = 0;
        const SHADER_FLOAT16 = 1 << 0;
        const SHADER_INT8 = 1 << 1;
        const SHADER_INT16 = 1 << 2;
        const COOPERATIVE_MATRIX = 1 << 3;
        const SUBGROUP_BASIC = 1 << 4;
        const SUBGROUP_VOTE = 1 << 5;
        const SUBGROUP_ARITHMETIC = 1 << 6;
        const SUBGROUP_BALLOT = 1 << 7;
        const SUBGROUP_SHUFFLE = 1 << 8;
        const SUBGROUP_CLUSTERED = 1 << 9;
        const SHADER_ATOMIC_INT64 = 1 << 10;
        const SHADER_ATOMIC_FLOAT = 1 << 11;
        const SHADER_INT64 = 1 << 12;
        const SHADER_INT64_ATOMICS = 1 << 13;
        const SHADER_FLOAT64 = 1 << 14;
        const SHADER_FLOAT64_ATOMICS = 1 << 15;
    }
}

impl CapabilitySet {
    /// Create an empty capability set
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a capability set from flags
    pub fn from_flags(flags: CapabilityFlags) -> Self {
        Self { flags }
    }

    /// Check if a capability is required
    pub fn has(&self, flag: CapabilityFlags) -> bool {
        self.flags.contains(flag)
    }

    /// Add a capability requirement
    pub fn with(mut self, flag: CapabilityFlags) -> Self {
        self.flags |= flag;
        self
    }
}

/// Data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
}
