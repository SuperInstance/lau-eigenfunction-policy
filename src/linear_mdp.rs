//! Linearly-solvable MDPs: HJB → linear via Hopf-Cole transform.
//!
//! In a linearly-solvable MDP, the transition kernel P and the passive (uncontrolled)
//! dynamics define a Dirichlet form. The Bellman equation becomes linear under
//! the desirability transform z = e^{-V/λ}.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A linearly-solvable MDP defined by a transition kernel and cost structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearlySolvableMDP {
    /// Number of states.
    pub n_states: usize,
    /// Number of actions.
    pub n_actions: usize,
    /// Passive transition kernel: P[a] is n_states × n_states.
    /// P[a](i, j) = probability of transitioning i → j under action a
    /// with passive dynamics.
    pub transitions: Vec<DMatrix<f64>>,
    /// State costs: c(s) for each state.
    pub state_costs: DVector<f64>,
    /// Temperature parameter λ (controls exploration vs exploitation).
    pub temperature: f64,
    /// Discount factor γ ∈ [0, 1).
    pub discount: f64,
    /// Terminal states (Dirichlet boundary conditions).
    pub terminal_states: Vec<usize>,
}

impl LinearlySolvableMDP {
    /// Create a new linearly-solvable MDP.
    pub fn new(
        n_states: usize,
        n_actions: usize,
        transitions: Vec<DMatrix<f64>>,
        state_costs: DVector<f64>,
        temperature: f64,
    ) -> Self {
        assert_eq!(transitions.len(), n_actions);
        for (a, p) in transitions.iter().enumerate() {
            assert_eq!(p.nrows(), n_states);
            assert_eq!(p.ncols(), n_states);
            // Verify stochasticity per row
            for i in 0..n_states {
                let row_sum: f64 = (0..n_states).map(|j| p[(i, j)]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-10,
                    "Row {} of transition[{}] sums to {} (expected 1.0)",
                    i,
                    a,
                    row_sum
                );
            }
        }
        assert_eq!(state_costs.len(), n_states);

        Self {
            n_states,
            n_actions,
            transitions,
            state_costs,
            temperature,
            discount: 1.0,
            terminal_states: Vec::new(),
        }
    }

    /// Build the passive dynamics matrix (average over actions with uniform weights).
    pub fn passive_dynamics(&self) -> DMatrix<f64> {
        let n = self.n_states;
        let mut p = DMatrix::zeros(n, n);
        for pa in &self.transitions {
            p += pa;
        }
        p / (self.n_actions as f64)
    }

    /// Build the desirability-weighted transition operator.
    /// This is the key operator: L = P * diag(exp(-c/λ))
    /// where P is the passive dynamics and c is the state cost.
    ///
    /// The principal eigenfunction of L gives the desirability function z.
    pub fn desirability_operator(&self) -> DMatrix<f64> {
        let p = self.passive_dynamics();
        let weights = self.desirability_weights();
        let d = DMatrix::from_diagonal(&weights);
        &p * &d
    }

    /// Compute desirability weights: exp(-c(s)/λ) for each state.
    pub fn desirability_weights(&self) -> DVector<f64> {
        self.state_costs
            .map(|c| (-c / self.temperature).exp())
    }

    /// Build the Dirichlet-form operator (with absorbing boundary at terminal states).
    /// Terminal states get zero rows (Dirichlet boundary condition).
    pub fn dirichlet_operator(&self) -> DMatrix<f64> {
        let mut op = self.desirability_operator();
        for &s in &self.terminal_states {
            for j in 0..self.n_states {
                op[(s, j)] = 0.0;
            }
        }
        op
    }

    /// Build action-weighted operator for a specific action.
    pub fn action_operator(&self, action: usize) -> DMatrix<f64> {
        let weights = self.desirability_weights();
        let d = DMatrix::from_diagonal(&weights);
        &self.transitions[action] * &d
    }

    /// Compute the optimal value function from desirability: V = -λ log(z).
    pub fn value_from_desirability(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|zi| {
            if zi > 0.0 {
                -self.temperature * zi.ln()
            } else {
                f64::INFINITY
            }
        })
    }

    /// Compute desirability from value function: z = exp(-V/λ).
    pub fn desirability_from_value(&self, v: &DVector<f64>) -> DVector<f64> {
        v.map(|vi| (-vi / self.temperature).exp())
    }

    /// Solve the linearly-solvable MDP by finding the principal eigenfunction.
    /// Returns (eigenvalue, desirability_function, value_function).
    pub fn solve(&self, max_iterations: usize, tolerance: f64) -> (f64, DVector<f64>, DVector<f64>) {
        let op = self.dirichlet_operator();
        let (eigenvalue, z) = crate::eigen::power_iteration(&op, max_iterations, tolerance);
        let v = self.value_from_desirability(&z);
        (eigenvalue, z, v)
    }

    /// Create a simple grid-world MDP.
    pub fn gridworld(size: usize, goal: usize, temperature: f64) -> Self {
        let n = size * size;
        // Uniform random transitions as passive dynamics
        let mut transitions = Vec::new();
        // 4 actions: up, down, left, right
        for _ in 0..4 {
            let mut p = DMatrix::zeros(n, n);
            for i in 0..n {
                let row = i / size;
                let col = i % size;
                let mut neighbors = Vec::new();
                if row > 0 { neighbors.push((row - 1) * size + col); }
                if row + 1 < size { neighbors.push((row + 1) * size + col); }
                if col > 0 { neighbors.push(row * size + (col - 1)); }
                if col + 1 < size { neighbors.push(row * size + (col + 1)); }
                let prob = 1.0 / neighbors.len() as f64;
                for &j in &neighbors {
                    p[(i, j)] = prob;
                }
            }
            transitions.push(p);
        }

        let mut costs = DVector::from_element(n, 1.0);
        costs[goal] = 0.0;

        let mut mdp = Self::new(n, 4, transitions, costs, temperature);
        mdp.terminal_states = vec![goal];
        mdp
    }

    /// Create a chain MDP (1D random walk with absorbing endpoint).
    pub fn chain(n: usize, temperature: f64) -> Self {
        let mut transitions = Vec::new();
        // 2 actions: left, right
        for _ in 0..2 {
            let mut p = DMatrix::zeros(n, n);
            for i in 0..n {
                if i == n - 1 {
                    // Terminal state
                    p[(i, i)] = 1.0;
                } else if i == 0 {
                    p[(i, i)] = 0.5;
                    p[(i, i + 1)] = 0.5;
                } else {
                    p[(i, i - 1)] = 0.5;
                    p[(i, i + 1)] = 0.5;
                }
            }
            transitions.push(p);
        }

        let mut costs = DVector::from_element(n, 1.0);
        costs[n - 1] = 0.0;

        let mut mdp = Self::new(n, 2, transitions, costs, temperature);
        mdp.terminal_states = vec![n - 1];
        mdp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mdp_creation() {
        let p = DMatrix::identity(3, 3);
        let mdp = LinearlySolvableMDP::new(
            3, 2,
            vec![p.clone(), p.clone()],
            DVector::from_vec(vec![1.0, 0.5, 0.0]),
            1.0,
        );
        assert_eq!(mdp.n_states, 3);
        assert_eq!(mdp.n_actions, 2);
        assert_eq!(mdp.temperature, 1.0);
    }

    #[test]
    fn test_passive_dynamics() {
        let p1 = DMatrix::identity(2, 2);
        let p2 = DMatrix::from_element(2, 2, 0.5);
        let mdp = LinearlySolvableMDP::new(2, 2, vec![p1, p2], DVector::zeros(2), 1.0);
        let passive = mdp.passive_dynamics();
        assert_relative_eq!(passive[(0, 0)], 0.75);
        assert_relative_eq!(passive[(0, 1)], 0.25);
        assert_relative_eq!(passive[(1, 0)], 0.25);
        assert_relative_eq!(passive[(1, 1)], 0.75);
    }

    #[test]
    fn test_desirability_weights() {
        let p = DMatrix::identity(2, 2);
        let mdp = LinearlySolvableMDP::new(
            2, 1,
            vec![p],
            DVector::from_vec(vec![0.0, 1.0]),
            1.0,
        );
        let w = mdp.desirability_weights();
        assert_relative_eq!(w[0], 1.0);
        assert_relative_eq!(w[1], std::f64::consts::E.powi(-1));
    }

    #[test]
    fn test_value_desirability_roundtrip() {
        let p = DMatrix::identity(3, 3);
        let mdp = LinearlySolvableMDP::new(
            3, 1,
            vec![p],
            DVector::zeros(3),
            2.0,
        );
        let v = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let z = mdp.desirability_from_value(&v);
        let v2 = mdp.value_from_desirability(&z);
        for i in 0..3 {
            assert_relative_eq!(v[i], v2[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_boundary() {
        let p = DMatrix::identity(3, 3);
        let mut mdp = LinearlySolvableMDP::new(
            3, 1,
            vec![p],
            DVector::zeros(3),
            1.0,
        );
        mdp.terminal_states = vec![2];
        let op = mdp.dirichlet_operator();
        // Row 2 should be all zeros (Dirichlet boundary)
        assert_relative_eq!(op[(2, 0)], 0.0);
        assert_relative_eq!(op[(2, 1)], 0.0);
        assert_relative_eq!(op[(2, 2)], 0.0);
    }

    #[test]
    fn test_desirability_operator() {
        let p = DMatrix::identity(2, 2);
        let mdp = LinearlySolvableMDP::new(
            2, 1,
            vec![p],
            DVector::from_vec(vec![0.0, 0.0]),
            1.0,
        );
        let op = mdp.desirability_operator();
        // With zero costs, weights are all 1, so operator = P
        assert_relative_eq!(op[(0, 0)], 1.0);
        assert_relative_eq!(op[(1, 1)], 1.0);
    }

    #[test]
    fn test_gridworld_creation() {
        let mdp = LinearlySolvableMDP::gridworld(3, 8, 1.0);
        assert_eq!(mdp.n_states, 9);
        assert_eq!(mdp.n_actions, 4);
        assert_eq!(mdp.terminal_states, vec![8]);
    }

    #[test]
    fn test_chain_creation() {
        let mdp = LinearlySolvableMDP::chain(5, 1.0);
        assert_eq!(mdp.n_states, 5);
        assert_eq!(mdp.n_actions, 2);
        assert_eq!(mdp.terminal_states, vec![4]);
    }

    #[test]
    fn test_chain_solve() {
        let mdp = LinearlySolvableMDP::chain(5, 0.5);
        let (eigenvalue, z, v) = mdp.solve(1000, 1e-12);
        // Eigenvalue should be positive and < 1 for absorbing chain
        assert!(eigenvalue > 0.0);
        // Desirability should be non-negative
        for i in 0..5 {
            assert!(z[i] >= 0.0);
        }
        // At least some interior states should have positive desirability
        assert!(z[0] > 0.0 || z[1] > 0.0);
        // Non-terminal states should have finite value
        for i in 0..4 {
            if z[i] > 0.0 {
                assert!(v[i].is_finite());
            }
        }
    }

    #[test]
    fn test_solve_gridworld() {
        let mdp = LinearlySolvableMDP::gridworld(3, 8, 1.0);
        let (eigenvalue, z, v) = mdp.solve(1000, 1e-12);
        assert!(eigenvalue > 0.0);
        // Desirability should be non-negative
        for i in 0..9 {
            assert!(z[i] >= 0.0);
        }
        // Some states should have positive desirability
        assert!(z.iter().any(|&zi| zi > 0.0));
    }

    #[test]
    fn test_action_operator() {
        let p = DMatrix::identity(2, 2);
        let mdp = LinearlySolvableMDP::new(
            2, 1,
            vec![p],
            DVector::from_vec(vec![0.0, 1.0]),
            1.0,
        );
        let op = mdp.action_operator(0);
        assert_relative_eq!(op[(0, 0)], 1.0);
        assert_relative_eq!(op[(0, 1)], 0.0);
        assert_relative_eq!(op[(1, 0)], 0.0);
        assert_relative_eq!(op[(1, 1)], (-1.0f64).exp());
    }

    #[test]
    #[should_panic]
    fn test_invalid_transition_kernel() {
        let bad = DMatrix::from_element(2, 2, 0.3); // rows don't sum to 1
        let _ = LinearlySolvableMDP::new(
            2, 1,
            vec![bad],
            DVector::zeros(2),
            1.0,
        );
    }

    #[test]
    fn test_temperature_effect() {
        let p = DMatrix::identity(3, 3);
        let costs = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let mdp1 = LinearlySolvableMDP::new(3, 1, vec![p.clone()], costs.clone(), 0.1);
        let mdp2 = LinearlySolvableMDP::new(3, 1, vec![p], costs, 10.0);
        let w1 = mdp1.desirability_weights();
        let w2 = mdp2.desirability_weights();
        // Lower temperature → more peaked desirability
        assert!(w1[2] < w2[2]); // Low temp: high cost state less desirable
    }
}
