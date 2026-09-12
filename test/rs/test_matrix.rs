//! Schema-generated matrix oracles.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "matrix/test_blas.gen.rs"]
mod blas;
#[path = "matrix/test_broadcast.rs"]
mod broadcast;
#[path = "matrix/test_copy.rs"]
mod copy;
#[path = "matrix/test_elemwise.gen.rs"]
mod elemwise;
#[path = "matrix/test_index.rs"]
mod index;
#[path = "matrix/test_reduce.gen.rs"]
mod reduce;
#[path = "matrix/test_rng.rs"]
mod rng;
