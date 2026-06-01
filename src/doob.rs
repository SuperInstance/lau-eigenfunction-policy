//! Doob h-transform: conditioned heat flow = optimal policy.
//!
//! The Doob h-transform conditions a Markov process on reaching a target.
//! When h is the ground state eigenfunction, the conditioned process IS the
//! optimal policy. This is the deep connection between conditioning and control.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Doob h-transform: transforms the passive dynamics into the optimal policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoobHTransform {
    /// The ground state eigenfunction h (desirability).
    pub h: DVector<f64>,
    /// Number of states.
    pub n_states: usize,
}

impl DoobHTransform {
    /// Create a Doob h-transform from the ground state eigenfunction.
    pub fn new(h: DVector<f64>) -> Self {
        let n = h.len();
        for (i, &hi) in h.iter().enumerate() {
            assert!(hi >= 0.0, "Ground state must be non-negative (state {} = {})", i, hi);
        }
        Self { h: h.clone(), n_states: n }
    }

    /// Apply the h-transform to a transition kernel.
    /// The conditioned kernel is: P̃(i,j) = P(i,j) h(j) / (Ph)(i)
    pub fn transform(&self, kernel: &DMatrix<f64>) -> DMatrix<f64> {
        let ph = kernel * &self.h;
        let mut result = DMatrix::zeros(self.n_states, self.n_states);

        for i in 0..self.n_states {
            let normalizer = ph[i];
            if normalizer.abs() < 1e-30 {
                // State i is inaccessible or terminal
                continue;
            }
            for j in 0..self.n_states {
                result[(i, j)] = kernel[(i, j)] * self.h[j] / normalizer;
            }
        }

        result
    }

    /// Apply the h-transform to get action-conditioned transition.
    pub fn transform_action(&self, action_kernel: &DMatrix<f64>) -> DMatrix<f64> {
        self.transform(action_kernel)
    }

    /// Compute the optimal policy: probability of each action given state.
    /// π(a|s) = P_a z / Σ_a' P_a' z
    pub fn optimal_policy(
        &self,
        action_kernels: &[DMatrix<f64>],
    ) -> DMatrix<f64> {
        let n_actions = action_kernels.len();
        let n = self.n_states;

        // Compute P_a * z for each action
        let paz: Vec<DVector<f64>> = action_kernels
            .iter()
            .map(|pa| pa * &self.h)
            .collect();

        // Sum over actions: total desirability
        let mut total = DVector::zeros(n);
        for paz_a in &paz {
            total += paz_a;
        }

        // Policy: π(a|s) = (P_a z)(s) / Σ_a (P_a z)(s)
        let mut policy = DMatrix::zeros(n, n_actions);
        for (a, paz_a) in paz.iter().enumerate() {
            for s in 0..n {
                if total[s].abs() > 1e-30 {
                    policy[(s, a)] = paz_a[s] / total[s];
                }
            }
        }

        policy
    }

    /// Compute the policy as a deterministic (greedy) map.
    /// Returns the best action for each state.
    pub fn greedy_policy(&self, action_kernels: &[DMatrix<f64>]) -> Vec<usize> {
        let policy = self.optimal_policy(action_kernels);
        let n_actions = action_kernels.len();
        (0..self.n_states)
            .map(|s| {
                let mut best = 0;
                let mut best_val = policy[(s, 0)];
                for a in 1..n_actions {
                    if policy[(s, a)] > best_val {
                        best_val = policy[(s, a)];
                        best = a;
                    }
                }
                best
            })
            .collect()
    }

    /// Compute the Doob-conditioned transition kernel (h-transformed).
    /// This is the transition kernel under the optimal policy.
    pub fn conditioned_kernel(
        &self,
        passive_kernel: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        self.transform(passive_kernel)
    }

    /// Compute the Feynman-Kac representation: expected value of
    /// a function along the conditioned process.
    pub fn feynman_kac(
        &self,
        kernel: &DMatrix<f64>,
        reward: &DVector<f64>,
        discount: f64,
        steps: usize,
    ) -> DVector<f64> {
        let conditioned = self.transform(kernel);
        let mut value = reward.clone();
        let mut power = conditioned.clone();

        for _ in 1..steps {
            value += discount * (&power * reward);
            power = &power * &conditioned;
        }

        // Scale by h
        let mut result = DVector::zeros(self.n_states);
        for i in 0..self.n_states {
            if self.h[i] > 1e-30 {
                result[i] = value[i];
            }
        }
        result
    }

    /// Verify that the h-transformed kernel is stochastic.
    pub fn verify_stochastic(&self, kernel: &DMatrix<f64>, tolerance: f64) -> bool {
        let conditioned = self.transform(kernel);
        for i in 0..self.n_states {
            let row_sum: f64 = (0..self.n_states).map(|j| conditioned[(i, j)]).sum();
            if self.h[i] > 1e-10 && (row_sum - 1.0).abs() > tolerance {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_h_transform_identity() {
        let h = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::from_vec(3, 3, vec![
            0.5, 0.0, 0.5,
            0.3, 0.4, 0.3,
            0.0, 0.5, 0.5,
        ]);
        let transformed = doob.transform(&p);
        // Rows should sum to 1 (for non-killed states)
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| transformed[(i, j)]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_optimal_policy() {
        let h = DVector::from_vec(vec![1.0, 2.0]);
        let doob = DoobHTransform::new(h);
        let p1 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let p2 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let policy = doob.optimal_policy(&[p1, p2]);
        // Policy should sum to 1 per state
        for s in 0..2 {
            let sum = policy[(s, 0)] + policy[(s, 1)];
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_greedy_policy() {
        let h = DVector::from_vec(vec![1.0, 10.0]);
        let doob = DoobHTransform::new(h);
        // Action 1 always goes to state 1 (high desirability)
        let p1 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let p2 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let greedy = doob.greedy_policy(&[p1, p2]);
        // State 0: action 1 goes to state 1 (h=10), should prefer action 1
        assert_eq!(greedy[0], 1);
    }

    #[test]
    fn test_conditioned_kernel_stochastic() {
        let h = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::from_vec(3, 3, vec![
            0.3, 0.3, 0.4,
            0.5, 0.3, 0.2,
            0.1, 0.6, 0.3,
        ]);
        assert!(doob.verify_stochastic(&p, 1e-10));
    }

    #[test]
    fn test_uniform_h_gives_original() {
        let h = DVector::from_vec(vec![1.0, 1.0]);
        let doob = DoobHTransform::new(h);
        // Column-major: vec![col0_row0, col0_row1, col1_row0, col1_row1]
        // For stochastic: row 0 = [0.7, 0.3], row 1 = [0.4, 0.6]
        // So column-major: col0=[0.7, 0.4], col1=[0.3, 0.6]
        let p = DMatrix::from_vec(2, 2, vec![0.7, 0.4, 0.3, 0.6]);
        let transformed = doob.transform(&p);
        // With uniform h, transform should return original kernel
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(transformed[(i, j)], p[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_h_transform_biases_toward_high_h() {
        let h = DVector::from_vec(vec![1.0, 100.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
        let transformed = doob.transform(&p);
        // From state 0, should strongly prefer state 1 (h=100)
        assert!(transformed[(0, 1)] > transformed[(0, 0)]);
        assert!(transformed[(0, 1)] > 0.9);
    }

    #[test]
    fn test_feynman_kac() {
        let h = DVector::from_vec(vec![1.0, 1.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::identity(2, 2);
        let reward = DVector::from_vec(vec![1.0, 0.0]);
        let value = doob.feynman_kac(&p, &reward, 0.9, 10);
        assert!(value[0] > 0.0);
    }

    #[test]
    fn test_3x3_chain() {
        let h = DVector::from_vec(vec![0.5, 1.0, 2.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 0.5, 0.5,
            0.5, 0.0, 0.5,
            0.5, 0.5, 0.0,
        ]);
        let transformed = doob.transform(&p);
        // All rows should sum to ~1
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| transformed[(i, j)]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
        // State 0 should prefer state 2 (highest h)
        assert!(transformed[(0, 2)] > transformed[(0, 0)]);
    }

    #[test]
    #[should_panic]
    fn test_negative_h() {
        let h = DVector::from_vec(vec![1.0, -1.0]);
        DoobHTransform::new(h);
    }

    #[test]
    fn test_verify_stochastic() {
        let h = DVector::from_vec(vec![2.0, 3.0]);
        let doob = DoobHTransform::new(h);
        let p = DMatrix::from_vec(2, 2, vec![0.6, 0.4, 0.3, 0.7]);
        assert!(doob.verify_stochastic(&p, 1e-10));
    }
}
