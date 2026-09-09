//! Schema-generated matrix oracles.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "matrix/test_blas.gen.rs"]
mod blas;
#[path = "matrix/test_elemwise.gen.rs"]
mod elemwise;
