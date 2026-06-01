//! # lau-eigenfunction-policy
//!
//! Optimal RL policy as Dirichlet eigenfunction — solve RL via eigenvalues, not Bellman iteration.
//!
//! Implements Opus's Emergent Theorem A: the optimal RL policy is the principal eigenfunction
//! of the Dirichlet-form semigroup, and the policy is the Doob h-transform (ground state transform)
//! of the heat flow.
//!
//! Under the Hopf-Cole / desirability change of variables z = e^{-V/λ}, the nonlinear HJB
//! collapses to a linear eigenproblem. The optimal policy IS the ground state.

pub mod dirichlet;
pub mod doob;
pub mod eigen;
pub mod hopf_cole;
pub mod linear_mdp;
pub mod policy;
pub mod spectral;
pub mod wkb;

pub use dirichlet::DirichletLaplacian;
pub use doob::DoobHTransform;
pub use eigen::PowerIteration;
pub use hopf_cole::HopfColeTransform;
pub use linear_mdp::LinearlySolvableMDP;
pub use policy::EigenPolicy;
pub use spectral::SpectralAnalysis;
pub use wkb::WKBLimit;
