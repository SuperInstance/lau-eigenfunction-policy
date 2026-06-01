//! Principal eigenvalue computation via power iteration on Dirichlet Laplacian.
//!
//! The power method finds the dominant eigenvector (principal eigenfunction)
//! of the desirability operator. This IS the ground state — the optimal policy.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Power iteration solver for principal eigenvalue/eigenvector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerIteration {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl PowerIteration {
    /// Create a new power iteration solver.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Run power iteration on a matrix to find the principal eigenvalue/eigenvector.
    pub fn solve(&self, matrix: &DMatrix<f64>) -> (f64, DVector<f64>) {
        power_iteration(matrix, self.max_iterations, self.tolerance)
    }

    /// Run power iteration with a specific initial guess.
    pub fn solve_with_initial(
        &self,
        matrix: &DMatrix<f64>,
        initial: &DVector<f64>,
    ) -> (f64, DVector<f64>) {
        power_iteration_with_initial(matrix, initial, self.max_iterations, self.tolerance)
    }

    /// Estimate convergence rate from iteration history.
    /// The ratio |λ_{k+1} - λ_k| / |λ_k - λ_{k-1}| approaches |λ_2/λ_1|.
    pub fn estimate_convergence_ratio(
        &self,
        matrix: &DMatrix<f64>,
    ) -> f64 {
        let n = matrix.nrows();
        let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

        let mut prev_ratio = 0.0;
        let mut eigenvalue_old = 0.0;

        for iter in 0..self.max_iterations.min(200) {
            let w = matrix * &v;
            let norm = w.norm();
            if norm < 1e-30 {
                return 0.0;
            }
            let eigenvalue = v.dot(&w);
            v = w / norm;

            if iter >= 2 {
                let ratio = (eigenvalue - eigenvalue_old).abs();
                if prev_ratio > 0.0 {
                    let r = ratio / prev_ratio;
                    if iter > 10 {
                        return r;
                    }
                }
                prev_ratio = ratio;
            }
            eigenvalue_old = eigenvalue;
        }
        prev_ratio
    }
}

/// Standard power iteration: find principal eigenvalue and eigenvector.
pub fn power_iteration(
    matrix: &DMatrix<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> (f64, DVector<f64>) {
    let n = matrix.nrows();
    let initial = DVector::from_element(n, 1.0 / (n as f64).sqrt());
    power_iteration_with_initial(matrix, &initial, max_iterations, tolerance)
}

/// Power iteration with custom initial vector.
pub fn power_iteration_with_initial(
    matrix: &DMatrix<f64>,
    initial: &DVector<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> (f64, DVector<f64>) {
    let n = matrix.nrows();
    assert_eq!(matrix.ncols(), n);
    assert_eq!(initial.len(), n);

    let mut v = initial.clone();
    let mut eigenvalue = 0.0;

    for _ in 0..max_iterations {
        let w = matrix * &v;
        let norm = w.norm();
        if norm < 1e-30 {
            // Zero matrix or killed eigenvector
            return (0.0, v);
        }
        let new_eigenvalue = v.dot(&w);
        v = w / norm;

        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    // Ensure positive eigenvector (Perron-Frobenius: principal eigenvector is positive)
    if v.iter().any(|&x| x < 0.0) {
        v = -v;
    }

    (eigenvalue, v)
}

/// Shifted inverse iteration for finding eigenvalue closest to a target.
/// Useful for finding subdominant eigenvalues (spectral gap).
pub fn inverse_iteration(
    matrix: &DMatrix<f64>,
    shift: f64,
    max_iterations: usize,
    _tolerance: f64,
) -> (f64, DVector<f64>) {
    let n = matrix.nrows();
    let identity = DMatrix::identity(n, n);
    let shifted = matrix - identity.scale(shift);

    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

    for _ in 0..max_iterations {
        // Solve (A - σI)w = v
        let w = match shifted.clone().lu().solve(&v) {
            Some(w) => w,
            None => break,
        };
        let norm = w.norm();
        if norm < 1e-30 {
            break;
        }
        v = w / norm;
    }

    let eigenvalue = v.dot(&(matrix * &v));
    if v.iter().any(|&x| x < 0.0) && v.iter().all(|&x| x <= 0.0) {
        v = -v;
    }

    (eigenvalue, v)
}

/// Compute multiple eigenvalues using deflation.
pub fn deflated_eigenvalues(
    matrix: &DMatrix<f64>,
    num_eigenvalues: usize,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<(f64, DVector<f64>)> {
    let n = matrix.nrows();
    let count = num_eigenvalues.min(n);
    let mut results = Vec::with_capacity(count);
    let mut a = matrix.clone();

    for _ in 0..count {
        let (eigenvalue, eigenvector) = power_iteration(&a, max_iterations, tolerance);
        results.push((eigenvalue, eigenvector.clone()));

        // Deflate: remove this eigenvalue's contribution
        let v = &eigenvector;
        let vt = v.transpose();
        let outer = v * &vt;
        // Hotelling deflation: A' = A - λ vv^T (for symmetric-like)
        a = &a - outer.scale(eigenvalue);
    }

    results
}

/// Rayleigh quotient: gives eigenvalue estimate for a given eigenvector.
pub fn rayleigh_quotient(matrix: &DMatrix<f64>, v: &DVector<f64>) -> f64 {
    let av = matrix * v;
    let numerator = v.dot(&av);
    let denominator = v.dot(v);
    if denominator.abs() < 1e-30 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Residual norm: ||Av - λv||, measures eigenpair quality.
pub fn residual_norm(matrix: &DMatrix<f64>, eigenvalue: f64, eigenvector: &DVector<f64>) -> f64 {
    let av = matrix * eigenvector;
    let lv = eigenvector.scale(eigenvalue);
    (av - lv).norm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_matrix() {
        let i = DMatrix::identity(3, 3);
        let (eigenvalue, v) = power_iteration(&i, 100, 1e-12);
        assert_relative_eq!(eigenvalue, 1.0, epsilon = 1e-10);
        assert_relative_eq!(v.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_matrix() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0, 1.0]));
        let (eigenvalue, v) = power_iteration(&d, 100, 1e-12);
        assert_relative_eq!(eigenvalue, 3.0, epsilon = 1e-10);
        assert_relative_eq!(v[0].abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(v[1].abs(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(v[2].abs(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dominant_eigenvalue() {
        let m = DMatrix::from_vec(2, 2, vec![4.0, 1.0, 2.0, 3.0]);
        let (eigenvalue, _v) = power_iteration(&m, 1000, 1e-12);
        // Eigenvalues of [[4,2],[1,3]] are 5 and 2
        assert_relative_eq!(eigenvalue, 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_power_iteration_struct() {
        let solver = PowerIteration::new(1000, 1e-12);
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![5.0, 1.0]));
        let (eigenvalue, v) = solver.solve(&d);
        assert_relative_eq!(eigenvalue, 5.0, epsilon = 1e-10);
        assert_relative_eq!(v[0].abs(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_custom_initial() {
        let solver = PowerIteration::new(1000, 1e-12);
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0]));
        let initial = DVector::from_vec(vec![1.0, 0.0]);
        let (eigenvalue, _v) = solver.solve_with_initial(&d, &initial);
        assert_relative_eq!(eigenvalue, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rayleigh_quotient() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0]));
        let v = DVector::from_vec(vec![1.0, 0.0]);
        let rq = rayleigh_quotient(&d, &v);
        assert_relative_eq!(rq, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_norm() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0]));
        let v = DVector::from_vec(vec![1.0, 0.0]);
        let res = residual_norm(&d, 3.0, &v);
        assert_relative_eq!(res, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_nonzero() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0]));
        let v = DVector::from_vec(vec![0.6, 0.8]);
        let res = residual_norm(&d, 3.0, &v);
        assert!(res > 0.1); // Not an eigenvector for λ=3
    }

    #[test]
    fn test_deflated_eigenvalues() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![5.0, 3.0, 1.0]));
        let eigs = deflated_eigenvalues(&d, 3, 1000, 1e-12);
        assert_eq!(eigs.len(), 3);
        assert_relative_eq!(eigs[0].0, 5.0, epsilon = 1e-6);
        assert_relative_eq!(eigs[1].0, 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_positive_eigenvector() {
        let m = DMatrix::from_vec(2, 2, vec![0.8, 0.2, 0.3, 0.7]);
        let (_, v) = power_iteration(&m, 1000, 1e-12);
        // Principal eigenvector of positive matrix should be positive (Perron-Frobenius)
        assert!(v[0] >= 0.0);
        assert!(v[1] >= 0.0);
    }

    #[test]
    fn test_zero_matrix() {
        let z = DMatrix::zeros(3, 3);
        let (eigenvalue, _) = power_iteration(&z, 100, 1e-12);
        assert_relative_eq!(eigenvalue, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_ratio() {
        let solver = PowerIteration::new(200, 1e-12);
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.5, 0.1]));
        let ratio = solver.estimate_convergence_ratio(&d);
        // Should be close to |λ_2/λ_1| = 0.5
        assert!(ratio < 1.0);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_inverse_iteration() {
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0, 2.0]));
        let (eigenvalue, _) = inverse_iteration(&d, 2.1, 200, 1e-10);
        // Should find eigenvalue closest to shift 2.1, which is 2.0
        assert_relative_eq!(eigenvalue, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_larger_matrix() {
        let n = 20;
        let mut m = DMatrix::zeros(n, n);
        // Symmetric positive matrix
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = 1.0 / (1.0 + ((i as f64) - (j as f64)).abs());
            }
        }
        let (eigenvalue, v) = power_iteration(&m, 10000, 1e-14);
        assert!(eigenvalue > 0.0);
        assert_relative_eq!(v.norm(), 1.0, epsilon = 1e-10);
        let res = residual_norm(&m, eigenvalue, &v);
        assert!(res < 1e-6, "Residual too large: {}", res);
    }
}
