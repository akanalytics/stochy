#![doc=include_str!("../README.md")]
#![deny(
    future_incompatible,
    missing_docs,
    nonstandard_style,
    unsafe_op_in_unsafe_fn,
    unused,
    warnings,
    clippy::all,
    clippy::missing_safety_doc,
    clippy::undocumented_unsafe_blocks,
    rustdoc::broken_intra_doc_links,
    rustdoc::missing_crate_level_docs
)]
#![allow(clippy::empty_docs)]
#![cfg_attr(all(docsrs, not(doctest)), feature(doc_cfg, doc_auto_cfg))]

mod common;
mod rspsa;
mod spsa;

#[cfg(feature = "argmin")]
mod rspsa_argmin;

#[cfg(feature = "argmin")]
mod spsa_argmin;

#[cfg(feature = "argmin")]
#[cfg_attr(docsrs, doc(cfg(feature = "argmin")))]
pub use rspsa_argmin::RspsaSolverArgmin;

#[cfg(feature = "argmin")]
#[cfg_attr(docsrs, doc(cfg(feature = "argmin")))]
pub use spsa_argmin::SpsaSolverArgmin;

pub use rspsa::{RspsaParams, RspsaSolver};
pub use spsa::{SpsaParams, SpsaSolver};

use std::{error::Error, fmt::Display, sync::Arc};

/// The error type for the Stochy library.
#[derive(Debug, Clone)]
pub enum StochyError {
    /// Represents an error caused by an invalid hyperparameter.
    InvalidHyperparameter(String),
    /// Represents an error caused by calling the underlying objective function being solved
    ObjectiveFunction(Arc<dyn Error + Send + Sync + 'static>),
}
type BoxedError = Box<dyn Error + Send + Sync + 'static>;

impl From<BoxedError> for StochyError {
    fn from(err: BoxedError) -> Self {
        StochyError::ObjectiveFunction(Arc::from(err))
    }
}

impl Display for StochyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Error for StochyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self {
            Self::ObjectiveFunction(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

#[allow(dead_code)]
struct AssertSendSync<T: Send + Sync>(std::marker::PhantomData<T>);
const _: AssertSendSync<StochyError> = AssertSendSync(std::marker::PhantomData);
