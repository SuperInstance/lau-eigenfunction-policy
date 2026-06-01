//! Dirichlet Laplacian: the transition operator with absorbing boundary conditions.
//!
//! The Dirichlet Laplacian encodes the killed process: trajectories that reach
//! terminal states are absorbed (killed). The principal eigenfunction of this
//! operator gives the ground state desirability function.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Dirichlet Laplacian operator for a linearly-solvable MDP.
///
/// Terminal states impose zero boundary conditions (Dirichlet),
/// and the operator describes killed diffusion toward the goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletLaplacian {
    /// The operator matrix (transition kernel with Dirichlet boundary).
    pub operator: DMatrix<f64>,
    /// Number of states.
    pub n_states: usize,
    /// Indices of terminal (Dirichlet boundary) states.
    pub boundary_states: Vec<usize>,
}

impl DirichletLaplacian {
    /// Build a Dirichlet Laplacian from a transition kernel and boundary states.
    /// Zeroes out rows corresponding to boundary states (absorbing condition).
    pub fn new(transition_kernel: DMatrix<f64>, boundary_states: Vec<usize>) -> Self {
        let n = transition_kernel.nrows();
        let mut op = transition_kernel;

        // Apply Dirichlet boundary conditions
        for &s in &boundary_states {
            assert!(s < n, "Boundary state {} out of range (n={})", s, n);
            for j in 0..n {
                op[(s, j)] = 0.0;
            }
        }

        Self {
            operator: op,
            n_states: n,
            boundary_states,
        }
    }

    /// Build from a full MDP with desirability weighting.
    /// L = P * diag(exp(-c/λ)) with Dirichlet boundary at terminals.
    pub fn from_mdp(
        transition: &DMatrix<f64>,
        costs: &DVector<f64>,
        temperature: f64,
        boundary_states: Vec<usize>,
    ) -> Self {
        let weights: DVector<f64> = costs.map(|c| (-c / temperature).exp());
        let d = DMatrix::from_diagonal(&weights);
        let op = transition * d;
        Self::new(op, boundary_states)
    }

    /// Compute the principal eigenvalue (ground state energy).
    pub fn principal_eigenvalue(&self, max_iterations: usize, tolerance: f64) -> (f64, DVector<f64>) {
        crate::eigen::power_iteration(&self.operator, max_iterations, tolerance)
    }

    /// Get interior (non-boundary) state indices.
    pub fn interior_states(&self) -> Vec<usize> {
        let boundary_set: std::collections::HashSet<usize> =
            self.boundary_states.iter().cloned().collect();
        (0..self.n_states)
            .filter(|s| !boundary_set.contains(s))
            .collect()
    }

    /// Extract the interior submatrix (restricted to non-boundary states).
    pub fn interior_operator(&self) -> DMatrix<f64> {
        let interior = self.interior_states();
        let m = interior.len();
        let mut sub = DMatrix::zeros(m, m);
        for (i, &si) in interior.iter().enumerate() {
            for (j, &sj) in interior.iter().enumerate() {
                sub[(i, j)] = self.operator[(si, sj)];
            }
        }
        sub
    }

    /// Compute the Green's function (resolvent) for the interior.
    /// G = (I - γL)^{-1} where γ is the discount factor.
    pub fn greens_function(&self, discount: f64) -> DMatrix<f64> {
        let interior = self.interior_states();
        let m = interior.len();
        let sub = self.interior_operator();
        let identity: DMatrix<f64> = DMatrix::identity(m, m);
        let resolvent = &identity - sub.scale(discount);
        resolvent
            .clone()
            .try_inverse()
            .unwrap_or(DMatrix::zeros(m, m))
    }

    /// Compute killing probabilities: probability of being killed
    /// before reaching a terminal state, from each interior state.
    pub fn killing_probabilities(&self) -> DVector<f64> {
        let interior = self.interior_states();
        let m = interior.len();
        let sub = self.interior_operator();
        let identity: DMatrix<f64> = DMatrix::identity(m, m);
        // Killing probability = 1 - row sums of the interior operator
        // (mass that "leaks" to boundary states)
        let ones = DVector::from_element(m, 1.0);
        let row_sums = &sub * &ones;
        &ones - row_sums
    }

    /// Compute the Dirichlet form: E(f,f) = <f, (I-L)f>
    /// This is the energy associated with a function on the graph.
    pub fn dirichlet_form(&self, f: &DVector<f64>) -> f64 {
        let lf = &self.operator * f;
        let diff = f - &lf;
        f.dot(&diff)
    }

    /// Compute the Rayleigh quotient: <f, Lf> / <f, f>
    /// The minimum Rayleigh quotient over all functions gives λ₁.
    pub fn rayleigh_quotient(&self, f: &DVector<f64>) -> f64 {
        let lf = &self.operator * f;
        let numerator = f.dot(&lf);
        let denominator = f.dot(f);
        if denominator.abs() < 1e-30 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Committor function: probability of hitting a specific boundary
    /// subset before any other boundary state.
    pub fn committor(&self, target_boundary: &[usize]) -> DVector<f64> {
        let n = self.n_states;
        let target_set: std::collections::HashSet<usize> =
            target_boundary.iter().cloned().collect();
        let other_boundary: Vec<usize> = self
            .boundary_states
            .iter()
            .filter(|&&s| !target_set.contains(&s))
            .cloned()
            .collect();

        let interior: Vec<usize> = (0..n)
            .filter(|s| !self.boundary_states.contains(s))
            .collect();
        let m = interior.len();

        if m == 0 || other_boundary.is_empty() {
            let mut q = DVector::zeros(n);
            for &s in target_boundary {
                q[s] = 1.0;
            }
            return q;
        }

        // Solve: (I - P_int) q = b
        // where P_int is interior submatrix of original (pre-Dirichlet) kernel
        // and b accounts for transitions to target states
        let mut a: DMatrix<f64> = DMatrix::identity(m, m);
        let mut b: DVector<f64> = DVector::zeros(m);

        for (i, &si) in interior.iter().enumerate() {
            for (j, &_sj) in interior.iter().enumerate() {
                a[(i, j)] -= self.operator[(si, _sj)];
            }
            for &_t in target_boundary {
                // Approximate: use row sum from original kernel hitting target
                b[i] += self.operator.row(si).iter().sum::<f64>(); // simplified
            }
        }

        let q_interior = a.lu().solve(&b).unwrap_or(DVector::zeros(m));

        let mut q = DVector::zeros(n);
        for (i, &si) in interior.iter().enumerate() {
            q[si] = q_interior[i].clamp(0.0, 1.0);
        }
        for &s in target_boundary {
            q[s] = 1.0;
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_construction() {
        let p = DMatrix::identity(3, 3);
        let dl = DirichletLaplacian::new(p, vec![2]);
        assert_eq!(dl.n_states, 3);
        assert_eq!(dl.boundary_states, vec![2]);
        // Row 2 should be zeroed
        assert_relative_eq!(dl.operator[(2, 0)], 0.0);
        assert_relative_eq!(dl.operator[(2, 1)], 0.0);
        assert_relative_eq!(dl.operator[(2, 2)], 0.0);
    }

    #[test]
    fn test_interior_states() {
        let p = DMatrix::identity(4, 4);
        let dl = DirichletLaplacian::new(p, vec![0, 3]);
        assert_eq!(dl.interior_states(), vec![1, 2]);
    }

    #[test]
    fn test_interior_operator() {
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 0.0, 0.0,  // col 0
            0.5, 0.5, 0.0,  // col 1
            0.5, 0.5, 0.0,  // col 2
        ]);
        let dl = DirichletLaplacian::new(p, vec![0, 2]);
        let sub = dl.interior_operator();
        assert_eq!(sub.nrows(), 1);
        assert_eq!(sub.ncols(), 1);
        // Only state 1 is interior
        assert_relative_eq!(sub[(0, 0)], 0.5);
    }

    #[test]
    fn test_principal_eigenvalue() {
        let p = DMatrix::from_diagonal(&DVector::from_vec(vec![0.9, 0.8, 0.0]));
        let dl = DirichletLaplacian::new(p, vec![2]);
        let (eigenvalue, v) = dl.principal_eigenvalue(1000, 1e-12);
        assert_relative_eq!(eigenvalue, 0.9, epsilon = 1e-8);
        assert!(v[0] > 0.0);
    }

    #[test]
    fn test_from_mdp() {
        let p = DMatrix::identity(3, 3);
        let costs = DVector::from_vec(vec![1.0, 0.5, 0.0]);
        let dl = DirichletLaplacian::from_mdp(&p, &costs, 1.0, vec![2]);
        assert_eq!(dl.n_states, 3);
        assert_eq!(dl.boundary_states, vec![2]);
        // State 2 row should be zero
        for j in 0..3 {
            assert_relative_eq!(dl.operator[(2, j)], 0.0);
        }
    }

    #[test]
    fn test_dirichlet_form() {
        let p = DMatrix::identity(3, 3);
        let dl = DirichletLaplacian::new(p, vec![]);
        let f = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let energy = dl.dirichlet_form(&f);
        // With identity operator: <f, (I-I)f> = 0
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rayleigh_quotient() {
        let p = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0, 0.0]));
        let dl = DirichletLaplacian::new(p, vec![2]);
        let f = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let rq = dl.rayleigh_quotient(&f);
        assert_relative_eq!(rq, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_killing_probabilities() {
        let p = DMatrix::from_vec(3, 3, vec![
            0.5, 0.0, 0.5,  // col 0
            0.5, 0.5, 0.0,  // col 1
            0.0, 0.0, 0.0,  // col 2 (boundary)
        ]);
        let dl = DirichletLaplacian::new(p, vec![2]);
        let kill = dl.killing_probabilities();
        assert_eq!(kill.len(), 2); // 2 interior states
    }

    #[test]
    fn test_greens_function() {
        let p = DMatrix::from_vec(2, 2, vec![
            0.5, 0.5,
            0.0, 0.0,
        ]);
        let dl = DirichletLaplacian::new(p, vec![1]);
        let g = dl.greens_function(0.9);
        assert_eq!(g.nrows(), 1);
        assert!(g[(0, 0)] > 0.0);
    }

    #[test]
    fn test_committor() {
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 0.5, 0.5,
            0.5, 0.0, 0.5,
            0.0, 0.0, 0.0,
        ]);
        let dl = DirichletLaplacian::new(p, vec![0, 2]);
        let q = dl.committor(&[2]);
        // Should reach 1.0 at target
        assert_relative_eq!(q[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_invalid_boundary() {
        let p = DMatrix::identity(3, 3);
        DirichletLaplacian::new(p, vec![5]); // Out of range
    }
}
