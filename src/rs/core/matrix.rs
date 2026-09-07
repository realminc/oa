//! Matrix - numeric dtype, shape, strides, offset, and storage

use crate::error::Result;

/// Matrix represents numeric storage with explicit shape and dtype semantics.
pub struct Matrix {
    // TODO: Add storage, shape, dtype, strides, offset
}

impl Matrix {
    /// Create a matrix filled with ones
    pub fn ones(_shape: impl Into<[usize; 2]>) -> Result<Self> {
        todo!("Matrix::ones")
    }

    /// Get the shape of the matrix
    pub fn shape(&self) -> &[usize] {
        &[]
    }
}
