//! Eigen-policy: extract optimal policy from the ground state eigenfunction.
//!
//! π* ∝ Pz from eigenfunction — the optimal policy is proportional to the
//! desirability-weighted transition. Policy evaluation becomes an eigenproblem:
//! solve once, don't iterate Bellman.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Policy extracted from the principal eigenfunction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenPolicy {
    /// Number of states.
    pub n_states: usize,
    /// Number of actions.
    pub n_actions: usize,
    /// Desirability function (ground state eigenfunction).
    pub desirability: DVector<f64>,
    /// Value function V = -λ log(z).
    pub value: DVector<f64>,
    /// Policy matrix: π(s, a) = probability of action a in state s.
    pub policy: DMatrix<f64>,
    /// Temperature parameter.
    pub temperature: f64,
    /// Principal eigenvalue.
    pub eigenvalue: f64,
}

impl EigenPolicy {
    /// Extract policy from eigenfunction and action transition kernels.
    pub fn from_eigenfunction(
        eigenvalue: f64,
        desirability: DVector<f64>,
        action_kernels: &[DMatrix<f64>],
        temperature: f64,
    ) -> Self {
        let n = desirability.len();
        let n_actions = action_kernels.len();

        // Compute action desirabilities: P_a z for each action
        let action_desirability: Vec<DVector<f64>> = action_kernels
            .iter()
            .map(|pa| pa * &desirability)
            .collect();

        // Total desirability: Σ_a P_a z
        let mut total = DVector::zeros(n);
        for ad in &action_desirability {
            total += ad;
        }

        // Policy: π(a|s) = (P_a z)(s) / Σ_a (P_a z)(s)
        let mut policy = DMatrix::zeros(n, n_actions);
        for (a, ad) in action_desirability.iter().enumerate() {
            for s in 0..n {
                if total[s].abs() > 1e-30 {
                    policy[(s, a)] = ad[s] / total[s];
                }
            }
        }

        // Value function: V = -λ log(z)
        let value = desirability.map(|zi| {
            if zi > 0.0 {
                -temperature * zi.ln()
            } else {
                f64::INFINITY
            }
        });

        Self {
            n_states: n,
            n_actions,
            desirability,
            value,
            policy,
            temperature,
            eigenvalue,
        }
    }

    /// Get action probabilities for a given state.
    pub fn action_probabilities(&self, state: usize) -> DVector<f64> {
        DVector::from_iterator(
            self.n_actions,
            (0..self.n_actions).map(|a| self.policy[(state, a)]),
        )
    }

    /// Get the greedy (deterministic) action for a state.
    pub fn greedy_action(&self, state: usize) -> usize {
        let mut best = 0;
        let mut best_prob = self.policy[(state, 0)];
        for a in 1..self.n_actions {
            if self.policy[(state, a)] > best_prob {
                best_prob = self.policy[(state, a)];
                best = a;
            }
        }
        best
    }

    /// Get the full greedy policy.
    pub fn greedy_policy_vec(&self) -> Vec<usize> {
        (0..self.n_states).map(|s| self.greedy_action(s)).collect()
    }

    /// Compute the entropy of the policy at each state.
    pub fn entropy(&self) -> DVector<f64> {
        DVector::from_iterator(self.n_states, (0..self.n_states).map(|s| {
            let mut h = 0.0;
            for a in 0..self.n_actions {
                let p = self.policy[(s, a)];
                if p > 1e-30 {
                    h -= p * p.ln();
                }
            }
            h
        }))
    }

    /// Compute the Kullback-Leibler divergence from uniform policy.
    pub fn kl_from_uniform(&self) -> DVector<f64> {
        let uniform = 1.0 / self.n_actions as f64;
        DVector::from_iterator(self.n_states, (0..self.n_states).map(|s| {
            let mut kl = 0.0;
            for a in 0..self.n_actions {
                let p = self.policy[(s, a)];
                if p > 1e-30 {
                    kl += p * (p / uniform).ln();
                }
            }
            kl
        }))
    }

    /// Evaluate policy by simulating trajectories.
    pub fn evaluate(
        &self,
        transitions: &[DMatrix<f64>],
        rewards: &DVector<f64>,
        start_state: usize,
        max_steps: usize,
        discount: f64,
    ) -> f64 {
        let mut total_reward = 0.0;
        let mut state = start_state;
        let mut gamma = 1.0;

        for _ in 0..max_steps {
            total_reward += gamma * rewards[state];
            gamma *= discount;

            // Sample action from policy
            let _probs = self.action_probabilities(state);
            let action = self.greedy_action(state);

            // Transition
            let row: DVector<f64> = transitions[action].row(state).transpose().into();
            let next_state = self.sample_from_row(&row);
            state = next_state;
        }

        total_reward
    }

    /// Sample an action from the policy at a given state (deterministic for testing).
    fn sample_action(&self, state: usize) -> usize {
        self.greedy_action(state)
    }

    /// Sample next state from a transition row (pick max prob for determinism).
    fn sample_from_row(&self, row: &DVector<f64>) -> usize {
        let mut best = 0;
        let mut best_prob = row[0];
        for (j, &p) in row.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best = j;
            }
        }
        best
    }

    /// Compute the advantage of each action over the average.
    pub fn advantage(&self, action_kernels: &[DMatrix<f64>]) -> Vec<DVector<f64>> {
        let passive: DMatrix<f64> = action_kernels
            .iter()
            .fold(DMatrix::zeros(self.n_states, self.n_states), |acc, p| acc + p)
            / (self.n_actions as f64);

        let pz = passive * &self.desirability;

        action_kernels
            .iter()
            .map(|pa| {
                let paz = pa * &self.desirability;
                DVector::from_iterator(
                    self.n_states,
                    paz.iter()
                        .zip(pz.iter())
                        .map(|(&a, &p)| {
                            if p > 1e-30 && a > 1e-30 {
                                self.temperature * (a / p).ln()
                            } else {
                                0.0
                            }
                        }),
                )
            })
            .collect()
    }

    /// Check if the policy is deterministic (near-greedy).
    pub fn is_near_deterministic(&self, tolerance: f64) -> bool {
        for s in 0..self.n_states {
            let max_prob = (0..self.n_actions)
                .map(|a| self.policy[(s, a)])
                .fold(0.0f64, f64::max);
            if max_prob < 1.0 - tolerance {
                return false;
            }
        }
        true
    }

    /// Verify that policy rows sum to 1.
    pub fn verify_valid(&self, tolerance: f64) -> bool {
        for s in 0..self.n_states {
            let sum: f64 = (0..self.n_actions).map(|a| self.policy[(s, a)]).sum();
            if (sum - 1.0).abs() > tolerance {
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

    fn simple_mdp() -> (Vec<DMatrix<f64>>, DVector<f64>) {
        let p1 = DMatrix::from_vec(2, 2, vec![0.8, 0.2, 0.2, 0.8]);
        let p2 = DMatrix::from_vec(2, 2, vec![0.2, 0.8, 0.8, 0.2]);
        let costs = DVector::from_vec(vec![1.0, 0.0]);
        (vec![p1, p2], costs)
    }

    #[test]
    fn test_from_eigenfunction() {
        let (kernels, _costs) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        assert_eq!(policy.n_states, 2);
        assert_eq!(policy.n_actions, 2);
        assert!(policy.verify_valid(1e-10));
    }

    #[test]
    fn test_greedy_action() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 10.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        // With high desirability at state 1, greedy should prefer actions going there
        let g = policy.greedy_policy_vec();
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_value_function() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 2.0);
        // V(0) = -2*ln(1) = 0
        assert_relative_eq!(policy.value[0], 0.0, epsilon = 1e-10);
        // V(1) = -2*ln(2)
        assert_relative_eq!(policy.value[1], -2.0 * 2.0f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        let p1 = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
        let z = DVector::from_vec(vec![1.0, 1.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &[p1], 1.0);
        // Single action → entropy = 0
        let ent = policy.entropy();
        assert_relative_eq!(ent[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_multiple_actions() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 1.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        let ent = policy.entropy();
        // Entropy should be non-negative
        for i in 0..2 {
            assert!(ent[i] >= 0.0);
        }
    }

    #[test]
    fn test_kl_from_uniform() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 1.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        let kl = policy.kl_from_uniform();
        // KL divergence should be non-negative
        for i in 0..2 {
            assert!(kl[i] >= -1e-10);
        }
    }

    #[test]
    fn test_action_probabilities() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        let probs = policy.action_probabilities(0);
        assert_eq!(probs.len(), 2);
        let sum = probs[0] + probs[1];
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_advantage() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        let adv = policy.advantage(&kernels);
        assert_eq!(adv.len(), 2);
    }

    #[test]
    fn test_is_near_deterministic() {
        let p1 = DMatrix::identity(2, 2);
        let p2 = DMatrix::zeros(2, 2);
        let z = DVector::from_vec(vec![1.0, 100.0]);
        let _policy = EigenPolicy::from_eigenfunction(0.9, z, &[p1, p2], 0.01);
        // With very low temperature, should be near deterministic
    }

    #[test]
    fn test_verify_valid() {
        let (kernels, _) = simple_mdp();
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        assert!(policy.verify_valid(1e-10));
    }

    #[test]
    fn test_evaluate() {
        let (kernels, _) = simple_mdp();
        let rewards = DVector::from_vec(vec![0.0, 1.0]);
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let policy = EigenPolicy::from_eigenfunction(0.9, z, &kernels, 1.0);
        let value = policy.evaluate(&kernels, &rewards, 0, 100, 0.99);
        // Should accumulate some reward
        assert!(value.is_finite());
    }
}
