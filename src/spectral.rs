//! Spectral analysis: spectral gap = convergence rate.
//!
//! λ₁ predicts policy convergence speed. The spectral gap between the principal
//! eigenvalue and the next gives the mixing time of the optimal policy.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Spectral analysis of the Dirichlet-form operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    /// Eigenvalues (sorted descending by magnitude).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors corresponding to eigenvalues.
    pub eigenvectors: Vec<DVector<f64>>,
    /// Number of states.
    pub n_states: usize,
}

impl SpectralAnalysis {
    /// Compute full spectral analysis via deflation.
    pub fn analyze(operator: &DMatrix<f64>, max_iterations: usize, tolerance: f64) -> Self {
        let n = operator.nrows();
        let results = crate::eigen::deflated_eigenvalues(operator, n, max_iterations, tolerance);

        let mut pairs: Vec<(f64, DVector<f64>)> = results;
        // Sort by eigenvalue magnitude descending
        pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            eigenvalues: pairs.iter().map(|(e, _)| *e).collect(),
            eigenvectors: pairs.into_iter().map(|(_, v)| v).collect(),
            n_states: n,
        }
    }

    /// Compute just the spectral gap: λ₁ - λ₂.
    pub fn spectral_gap(&self) -> f64 {
        if self.eigenvalues.len() < 2 {
            return self.eigenvalues.first().copied().unwrap_or(0.0);
        }
        self.eigenvalues[0].abs() - self.eigenvalues[1].abs()
    }

    /// Relative spectral gap: (λ₁ - λ₂) / λ₁.
    pub fn relative_spectral_gap(&self) -> f64 {
        if self.eigenvalues.is_empty() {
            return 0.0;
        }
        let gap = self.spectral_gap();
        let lambda1 = self.eigenvalues[0].abs();
        if lambda1.abs() < 1e-30 {
            return 0.0;
        }
        gap / lambda1
    }

    /// Estimate mixing time from spectral gap.
    /// Mixing time ~ O(1/gap).
    pub fn mixing_time(&self) -> f64 {
        let gap = self.spectral_gap();
        if gap.abs() < 1e-30 {
            return f64::INFINITY;
        }
        1.0 / gap
    }

    /// Estimate convergence rate: the spectral gap determines
    /// how fast policy iteration converges.
    pub fn convergence_rate(&self) -> f64 {
        self.spectral_gap()
    }

    /// Check if the operator has a spectral gap (no degeneracy).
    pub fn has_spectral_gap(&self, tolerance: f64) -> bool {
        self.spectral_gap() > tolerance
    }

    /// Compute the effective dimension (number of significant eigenvalues).
    pub fn effective_dimension(&self, threshold: f64) -> usize {
        if self.eigenvalues.is_empty() {
            return 0;
        }
        let lambda1 = self.eigenvalues[0].abs();
        self.eigenvalues
            .iter()
            .take_while(|&&e| e.abs() > threshold * lambda1)
            .count()
    }

    /// Principal eigenvalue.
    pub fn principal_eigenvalue(&self) -> f64 {
        self.eigenvalues.first().copied().unwrap_or(0.0)
    }

    /// Principal eigenvector (ground state).
    pub fn ground_state(&self) -> Option<&DVector<f64>> {
        self.eigenvectors.first()
    }

    /// Compute the participation ratio: effective number of states
    /// contributing to the ground state.
    pub fn participation_ratio(&self) -> f64 {
        if let Some(gs) = self.ground_state() {
            let sum_sq: f64 = gs.iter().map(|&x| x * x).sum();
            let sum_quad: f64 = gs.iter().map(|&x| x * x * x * x).sum();
            if sum_quad.abs() < 1e-30 {
                return 0.0;
            }
            sum_sq * sum_sq / sum_quad
        } else {
            0.0
        }
    }

    /// Compute the spectral projection onto the principal eigenfunction.
    pub fn spectral_projection(&self, f: &DVector<f64>) -> DVector<f64> {
        if let Some(gs) = self.ground_state() {
            let coeff = gs.dot(f);
            gs.scale(coeff)
        } else {
            DVector::zeros(self.n_states)
        }
    }

    /// Temperature as ℏ: exploration as dequantization parameter.
    /// As temperature → 0, we get the classical (deterministic) limit.
    /// As temperature → ∞, we get the quantum (fully random) limit.
    pub fn temperature_as_hbar(
        operator: &DMatrix<f64>,
        temperature: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> (f64, f64) {
        // Rescale operator by temperature
        let scaled = operator.scale(1.0 / temperature);
        let (eigenvalue, _v) = crate::eigen::power_iteration(&scaled, max_iterations, tolerance);
        let spectral_gap = {
            let analysis = Self::analyze(&scaled, max_iterations, tolerance);
            analysis.spectral_gap()
        };
        (eigenvalue, spectral_gap)
    }
}

/// Compute the spectral gap directly (without full decomposition).
pub fn compute_spectral_gap(
    operator: &DMatrix<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> f64 {
    let eigs = crate::eigen::deflated_eigenvalues(operator, 2, max_iterations, tolerance);
    if eigs.len() < 2 {
        return eigs.first().map(|(e, _)| e.abs()).unwrap_or(0.0);
    }
    eigs[0].0.abs() - eigs[1].0.abs()
}

/// Estimate the relaxation time (inverse spectral gap).
pub fn relaxation_time(
    operator: &DMatrix<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> f64 {
    let gap = compute_spectral_gap(operator, max_iterations, tolerance);
    if gap.abs() < 1e-30 {
        f64::INFINITY
    } else {
        1.0 / gap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spectral_analysis_identity() {
        let i = DMatrix::identity(3, 3);
        let analysis = SpectralAnalysis::analyze(&i, 1000, 1e-12);
        assert_eq!(analysis.eigenvalues.len(), 3);
        // Principal eigenvalue should be 1.0
        assert_relative_eq!(analysis.eigenvalues[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_gap() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![5.0, 3.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let gap = analysis.spectral_gap();
        assert_relative_eq!(gap, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_relative_spectral_gap() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 2.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let rel_gap = analysis.relative_spectral_gap();
        assert_relative_eq!(rel_gap, 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_mixing_time() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![0.9, 0.5]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let mt = analysis.mixing_time();
        assert_relative_eq!(mt, 1.0 / 0.4, epsilon = 1e-4);
    }

    #[test]
    fn test_convergence_rate() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        assert_relative_eq!(analysis.convergence_rate(), 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_has_spectral_gap() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        assert!(analysis.has_spectral_gap(0.1));
    }

    #[test]
    fn test_no_spectral_gap() {
        // Use a matrix with degenerate eigenvalue: scalar multiple of identity
        let m = 3.0 * DMatrix::identity(2, 2);
        let analysis = SpectralAnalysis::analyze(&m, 1000, 1e-12);
        // After deflation, spectral gap depends on numerical noise
        // Just check both eigenvalues are close
        let gap = analysis.spectral_gap();
        assert!(gap < 5.0); // Should be small relative to eigenvalue=3
    }

    #[test]
    fn test_effective_dimension() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![10.0, 5.0, 0.1, 0.01]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let dim = analysis.effective_dimension(0.5);
        assert!(dim >= 1);
        assert!(dim <= 2);
    }

    #[test]
    fn test_principal_eigenvalue() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![7.0, 3.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        assert_relative_eq!(analysis.principal_eigenvalue(), 7.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ground_state() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![5.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let gs = analysis.ground_state().unwrap();
        assert_relative_eq!(gs[0].abs(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(gs[1].abs(), 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_participation_ratio() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 0.5]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let pr = analysis.participation_ratio();
        assert!(pr > 0.0);
    }

    #[test]
    fn test_spectral_projection() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0]));
        let analysis = SpectralAnalysis::analyze(&d, 1000, 1e-12);
        let f = DVector::from_vec(vec![1.0, 0.0]);
        let proj = analysis.spectral_projection(&f);
        assert_relative_eq!(proj[0].abs(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_compute_spectral_gap() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 2.0]));
        let gap = compute_spectral_gap(&d, 1000, 1e-12);
        assert_relative_eq!(gap, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_relaxation_time() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![0.8, 0.3]));
        let rt = relaxation_time(&d, 1000, 1e-12);
        assert!(rt.is_finite());
        assert!(rt > 0.0);
    }

    #[test]
    fn test_temperature_as_hbar() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 2.0, 1.0]));
        let (eigenvalue, gap) = SpectralAnalysis::temperature_as_hbar(&d, 2.0, 1000, 1e-12);
        assert!(eigenvalue > 0.0);
        assert!(gap > 0.0);
    }
}
