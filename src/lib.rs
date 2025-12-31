//! XPBD (eXtended Position Based Dynamics) simulation library.

#![warn(clippy::pedantic)]
#![warn(missing_docs)]

pub mod constraint;
pub mod mesh;
pub mod xpbd;

#[cfg(feature = "raylib")]
pub mod viz;
