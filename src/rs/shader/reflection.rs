//! Shader reflection utilities

use crate::error::Result;

/// Reflected shader information
pub struct ShaderReflection {
    pub entry_points: Vec<EntryPoint>,
    pub parameters: Vec<Parameter>,
    pub bindings: Vec<Binding>,
}

/// Entry point information
pub struct EntryPoint {
    pub name: String,
    pub stage: Stage,
}

/// Shader stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Compute,
    Vertex,
    Fragment,
}

/// Parameter information
pub struct Parameter {
    pub name: String,
    pub type_name: String,
}

/// Binding information
pub struct Binding {
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: DescriptorType,
}

/// Descriptor type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorType {
    StorageBuffer,
    UniformBuffer,
    SampledImage,
    StorageImage,
}

/// Reflect shader metadata
pub fn reflect(_spirv: &[u8]) -> Result<ShaderReflection> {
    todo!("reflect")
}
