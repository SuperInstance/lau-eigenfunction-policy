//! WKB/semiclassical limit: Varadhan's lemma.
//!
//! In the semiclassical (low-temperature) limit, the value function satisfies
//! V(x) ≈ ½ d(x, goal)² where d is the distance in the state graph.
//! This is Varadhan's lemma: the log of the heat kernel gives the geodesic distance.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// WKB/semiclassical analysis of the eigenfunction policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WKBLimit {
    /// Temperature parameter (ℏ analog).
    pub temperature: f64,
    /// Number of states.
    pub n_states: usize,
}

impl WKBLimit {
    /// Create a new WKB analyzer.
    pub fn new(temperature: f64, n_states: usize) -> Self {
        Self { temperature, n_states }
    }

    /// Compute graph distances using Dijkstra's algorithm.
    pub fn graph_distances(
        transition: &DMatrix<f64>,
        source: usize,
    ) -> DVector<f64> {
        let n = transition.nrows();
        let mut dist = DVector::from_element(n, f64::INFINITY);
        dist[source] = 0.0;

        // Simple Bellman-Ford (since edge weights = -log of transition probs)
        for _ in 0..n {
            let mut updated = false;
            for i in 0..n {
                if dist[i].is_infinite() {
                    continue;
                }
                for j in 0..n {
                    if transition[(i, j)] > 1e-30 {
                        let weight = -transition[(i, j)].ln(); // edge weight
                        let new_dist = dist[i] + weight;
                        if new_dist < dist[j] {
                            dist[j] = new_dist;
                            updated = true;
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }

        dist
    }

    /// Compute shortest-path distances on a graph where edges are weighted
    /// by state costs (not transition probabilities).
    pub fn cost_distances(
        transition: &DMatrix<f64>,
        costs: &DVector<f64>,
        source: usize,
    ) -> DVector<f64> {
        let n = transition.nrows();
        let mut dist = DVector::from_element(n, f64::INFINITY);
        dist[source] = 0.0;

        for _ in 0..n {
            let mut updated = false;
            for i in 0..n {
                if dist[i].is_infinite() {
                    continue;
                }
                for j in 0..n {
                    if transition[(i, j)] > 1e-30 {
                        let new_dist = dist[i] + costs[j];
                        if new_dist < dist[j] {
                            dist[j] = new_dist;
                            updated = true;
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }

        dist
    }

    /// WKB approximation of the value function.
    /// In the semiclassical limit (λ → 0): V(x) ≈ ½ d(x, goal)²
    pub fn wkb_value_function(
        &self,
        transition: &DMatrix<f64>,
        goal: usize,
    ) -> DVector<f64> {
        let distances = Self::graph_distances(transition, goal);
        distances.map(|d| {
            if d.is_finite() {
                0.5 * d * d
            } else {
                f64::INFINITY
            }
        })
    }

    /// WKB approximation of the desirability function.
    /// z(x) ≈ exp(-½ d(x, goal)² / λ)
    pub fn wkb_desirability(
        &self,
        transition: &DMatrix<f64>,
        goal: usize,
    ) -> DVector<f64> {
        let distances = Self::graph_distances(transition, goal);
        distances.map(|d| {
            if d.is_finite() {
                (-0.5 * d * d / self.temperature).exp()
            } else {
                0.0
            }
        })
    }

    /// Compute the semiclassical limit: as temperature → 0,
    /// verify that V → ½ d².
    pub fn semiclassical_limit_error(
        &self,
        value_function: &DVector<f64>,
        transition: &DMatrix<f64>,
        goal: usize,
    ) -> f64 {
        let wkb_value = self.wkb_value_function(transition, goal);
        let n = self.n_states;
        let mut total_error = 0.0;
        let mut count = 0;

        for i in 0..n {
            if value_function[i].is_finite() && wkb_value[i].is_finite() {
                let error = (value_function[i] - wkb_value[i]).abs();
                total_error += error * error;
                count += 1;
            }
        }

        if count == 0 {
            f64::INFINITY
        } else {
            (total_error / count as f64).sqrt()
        }
    }

    /// Compute the Varadhan distance: -λ log(K_t(x,y)) → d(x,y)²
    /// where K_t is the heat kernel (transition matrix raised to power t).
    pub fn varadhan_distance(
        &self,
        transition: &DMatrix<f64>,
        state_x: usize,
        state_y: usize,
        time: usize,
    ) -> f64 {
        let mut kernel = transition.clone();
        for _ in 1..time {
            kernel = &kernel * transition;
        }

        let kxy = kernel[(state_x, state_y)];
        if kxy > 1e-30 {
            -self.temperature * kxy.ln()
        } else {
            f64::INFINITY
        }
    }

    /// Compute the full Varadhan distance matrix.
    pub fn varadhan_distance_matrix(
        &self,
        transition: &DMatrix<f64>,
        time: usize,
    ) -> DMatrix<f64> {
        let n = transition.nrows();
        let mut kernel = transition.clone();
        for _ in 1..time {
            kernel = &kernel * transition;
        }

        let mut dist = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let k = kernel[(i, j)];
                dist[(i, j)] = if k > 1e-30 {
                    -self.temperature * k.ln()
                } else {
                    f64::INFINITY
                };
            }
        }
        dist
    }

    /// Check that the semiclassical limit holds:
    /// Varadhan distance → graph distance² as temperature → 0.
    pub fn verify_varadhan(
        &self,
        transition: &DMatrix<f64>,
        state_x: usize,
        state_y: usize,
        graph_dist: f64,
        time: usize,
        tolerance: f64,
    ) -> bool {
        let var_dist = self.varadhan_distance(transition, state_x, state_y, time);
        if var_dist.is_infinite() || graph_dist.is_infinite() {
            return true; // Can't verify for disconnected states
        }
        (var_dist - graph_dist * graph_dist).abs() < tolerance
    }

    /// Compute the classical (deterministic) policy via shortest paths.
    pub fn classical_policy(
        transition: &DMatrix<f64>,
        costs: &DVector<f64>,
        goal: usize,
    ) -> Vec<usize> {
        let n = transition.nrows();
        let mut dist = DVector::from_element(n, f64::INFINITY);
        dist[goal] = 0.0;
        let mut next_hop: Vec<usize> = (0..n).collect();

        // Backwards Dijkstra from goal
        for _ in 0..n {
            let mut updated = false;
            for i in 0..n {
                if dist[i].is_infinite() {
                    continue;
                }
                for j in 0..n {
                    if transition[(j, i)] > 1e-30 {
                        let new_dist = dist[i] + costs[j];
                        if new_dist < dist[j] {
                            dist[j] = new_dist;
                            next_hop[j] = i;
                            updated = true;
                        }
                    }
                }
            }
            if !updated {
                break;
            }
        }

        next_hop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_graph_distances_chain() {
        // 3-state chain: 0-1-2
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 0.5, 0.0,
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.0,
        ]);
        let dist = WKBLimit::graph_distances(&p, 0);
        assert_relative_eq!(dist[0], 0.0, epsilon = 1e-10);
        assert!(dist[1] > 0.0);
        assert!(dist[2] > dist[1]);
    }

    #[test]
    fn test_graph_distances_identity() {
        let p = DMatrix::identity(3, 3);
        let dist = WKBLimit::graph_distances(&p, 0);
        assert_relative_eq!(dist[0], 0.0, epsilon = 1e-10);
        // States 1, 2 unreachable (self-loops don't connect)
    }

    #[test]
    fn test_wkb_value_function() {
        let wkb = WKBLimit::new(1.0, 3);
        let p = DMatrix::from_vec(3, 3, vec![
            0.5, 0.5, 0.0,
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.5,
        ]);
        let v = wkb.wkb_value_function(&p, 0);
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-10);
        assert!(v[1] > 0.0);
    }

    #[test]
    fn test_wkb_desirability() {
        let wkb = WKBLimit::new(1.0, 3);
        let p = DMatrix::from_vec(3, 3, vec![
            0.5, 0.5, 0.0,
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.5,
        ]);
        let z = wkb.wkb_desirability(&p, 0);
        assert_relative_eq!(z[0], 1.0, epsilon = 1e-10);
        assert!(z[1] > 0.0 && z[1] < 1.0);
    }

    #[test]
    fn test_varadhan_distance() {
        let wkb = WKBLimit::new(1.0, 3);
        let p = DMatrix::identity(3, 3);
        let d = wkb.varadhan_distance(&p, 0, 0, 10);
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_varadhan_distance_matrix() {
        let wkb = WKBLimit::new(1.0, 2);
        let p = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
        let dm = wkb.varadhan_distance_matrix(&p, 5);
        assert_eq!(dm.nrows(), 2);
        assert_eq!(dm.ncols(), 2);
        // Diagonal should be smaller (closer to self)
        assert!(dm[(0, 0)] <= dm[(0, 1)]);
    }

    #[test]
    fn test_semiclassical_limit_error() {
        let wkb = WKBLimit::new(0.01, 3);
        let p = DMatrix::from_vec(3, 3, vec![
            0.5, 0.5, 0.0,
            0.5, 0.0, 0.5,
            0.0, 0.5, 0.5,
        ]);
        let v_wkb = wkb.wkb_value_function(&p, 0);
        let error = wkb.semiclassical_limit_error(&v_wkb, &p, 0);
        // WKB against itself should have zero error
        assert_relative_eq!(error, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_classical_policy() {
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 1.0, 0.0,
            0.5, 0.0, 0.5,
            0.0, 1.0, 0.0,
        ]);
        let costs = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let policy = WKBLimit::classical_policy(&p, &costs, 2);
        assert_eq!(policy.len(), 3);
        // State 0 should go to state 1 (toward goal 2)
        assert_eq!(policy[0], 1);
    }

    #[test]
    fn test_cost_distances() {
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ]);
        let costs = DVector::from_vec(vec![1.0, 1.0, 0.0]);
        let dist = WKBLimit::cost_distances(&p, &costs, 2);
        assert_relative_eq!(dist[2], 0.0, epsilon = 1e-10);
        assert!(dist[1] < dist[0]);
    }

    #[test]
    fn test_verify_varadhan() {
        let wkb = WKBLimit::new(1.0, 2);
        let p = DMatrix::identity(2, 2);
        // Self-distance is 0
        assert!(wkb.verify_varadhan(&p, 0, 0, 0.0, 10, 1.0));
    }

    #[test]
    fn test_wkb_with_mdp() {
        let wkb = WKBLimit::new(0.01, 3);
        let p = DMatrix::from_vec(3, 3, vec![
            0.0, 1.0, 0.0,
            0.5, 0.0, 0.5,
            0.0, 0.0, 1.0,
        ]);
        let v = wkb.wkb_value_function(&p, 2);
        // Goal should have value 0
        assert_relative_eq!(v[2], 0.0, epsilon = 1e-10);
        // State 1 should have finite value
        assert!(v[1].is_finite());
    }
}
