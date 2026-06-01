//! Hopf-Cole transform: z = e^{-V/λ}.
//!
//! The nonlinear HJB equation:
//!   λV = c + λ log(P · e^{-V/λ})
//!
//! Under the desirability change of variables z = e^{-V/λ}, this becomes:
//!   z = (1/μ) P · diag(e^{-c/λ}) · z
//!
//! which is a LINEAR eigenproblem. This is the Hopf-Cole (or logarithmic) transform.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Hopf-Cole transform utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopfColeTransform {
    /// Temperature parameter λ.
    pub temperature: f64,
}

impl HopfColeTransform {
    /// Create a new Hopf-Cole transform with given temperature.
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        Self { temperature }
    }

    /// Forward transform: value function → desirability function.
    /// z(s) = exp(-V(s)/λ)
    pub fn value_to_desirability(&self, v: &DVector<f64>) -> DVector<f64> {
        v.map(|vi| (-vi / self.temperature).exp())
    }

    /// Inverse transform: desirability → value function.
    /// V(s) = -λ log(z(s))
    pub fn desirability_to_value(&self, z: &DVector<f64>) -> DVector<f64> {
        z.map(|zi| {
            if zi > 0.0 {
                -self.temperature * zi.ln()
            } else {
                f64::INFINITY
            }
        })
    }

    /// Linearize the nonlinear Bellman/HJB equation.
    /// Takes the transition kernel P and state costs c, returns the
    /// linear operator L = P · diag(exp(-c/λ)).
    ///
    /// The eigenproblem Lz = μz replaces the nonlinear Bellman equation.
    pub fn linearize(
        &self,
        transition_kernel: &DMatrix<f64>,
        costs: &DVector<f64>,
    ) -> DMatrix<f64> {
        let weights: DVector<f64> = costs.map(|c| (-c / self.temperature).exp());
        let d = DMatrix::from_diagonal(&weights);
        transition_kernel * d
    }

    /// Linearize with action-dependent transitions.
    /// Returns operators for each action.
    pub fn linearize_actions(
        &self,
        transitions: &[DMatrix<f64>],
        costs: &DVector<f64>,
    ) -> Vec<DMatrix<f64>> {
        transitions
            .iter()
            .map(|p| self.linearize(p, costs))
            .collect()
    }

    /// Verify that the eigenfunction satisfies the linearized equation.
    /// Checks: Lz ≈ μz
    pub fn verify_eigenfunction(
        &self,
        operator: &DMatrix<f64>,
        z: &DVector<f64>,
        eigenvalue: f64,
        tolerance: f64,
    ) -> bool {
        let lz = operator * z;
        let expected = z.scale(eigenvalue);
        (lz - expected).norm() < tolerance * z.norm().max(1.0)
    }

    /// Compute the advantage function from desirability.
    /// The advantage of action a in state s is: log(P_a z / P z)
    /// where P_a is the action-a transition and P is the passive dynamics.
    pub fn advantage(
        &self,
        action_transitions: &[DMatrix<f64>],
        passive: &DMatrix<f64>,
        z: &DVector<f64>,
    ) -> Vec<DVector<f64>> {
        let pz = passive * z;
        action_transitions
            .iter()
            .map(|pa| {
                let paz = pa * z;
                DVector::from_iterator(
                    z.len(),
                    paz.iter()
                        .zip(pz.iter())
                        .map(|(&a, &p)| {
                            if p > 0.0 && a > 0.0 {
                                self.temperature * (a / p).ln()
                            } else {
                                0.0
                            }
                        }),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_forward_transform() {
        let hc = HopfColeTransform::new(1.0);
        let v = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let z = hc.value_to_desirability(&v);
        assert_relative_eq!(z[0], 1.0);
        assert_relative_eq!(z[1], (-1.0f64).exp());
        assert_relative_eq!(z[2], (-2.0f64).exp());
    }

    #[test]
    fn test_inverse_transform() {
        let hc = HopfColeTransform::new(2.0);
        let z = DVector::from_vec(vec![1.0, 0.5, 0.25]);
        let v = hc.desirability_to_value(&z);
        assert_relative_eq!(v[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(v[1], 2.0 * 2.0f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        let hc = HopfColeTransform::new(1.5);
        let v_orig = DVector::from_vec(vec![0.0, 0.5, 1.0, 3.0]);
        let z = hc.value_to_desirability(&v_orig);
        let v_back = hc.desirability_to_value(&z);
        for i in 0..v_orig.len() {
            assert_relative_eq!(v_orig[i], v_back[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zero_desirability_gives_infinity() {
        let hc = HopfColeTransform::new(1.0);
        let z = DVector::from_vec(vec![0.0, 1.0]);
        let v = hc.desirability_to_value(&z);
        assert!(v[0].is_infinite());
        assert!(v[1].is_finite());
    }

    #[test]
    fn test_linearize() {
        let hc = HopfColeTransform::new(1.0);
        let p = DMatrix::identity(2, 2);
        let costs = DVector::from_vec(vec![0.0, 0.0]);
        let l = hc.linearize(&p, &costs);
        // With zero costs, L = P
        assert_relative_eq!(l[(0, 0)], 1.0);
        assert_relative_eq!(l[(1, 1)], 1.0);
    }

    #[test]
    fn test_linearize_with_costs() {
        let hc = HopfColeTransform::new(1.0);
        let p = DMatrix::identity(2, 2);
        let costs = DVector::from_vec(vec![0.0, 1.0]);
        let l = hc.linearize(&p, &costs);
        assert_relative_eq!(l[(0, 0)], 1.0);
        assert_relative_eq!(l[(1, 1)], (-1.0f64).exp());
    }

    #[test]
    fn test_linearize_actions() {
        let hc = HopfColeTransform::new(1.0);
        let p1 = DMatrix::identity(2, 2);
        let p2 = DMatrix::from_element(2, 2, 0.5);
        let costs = DVector::zeros(2);
        let ops = hc.linearize_actions(&[p1, p2], &costs);
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn test_verify_eigenfunction_true() {
        let hc = HopfColeTransform::new(1.0);
        let l = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
        let z = DVector::from_vec(vec![1.0, 0.0]);
        assert!(hc.verify_eigenfunction(&l, &z, 2.0, 1e-10));
    }

    #[test]
    fn test_verify_eigenfunction_false() {
        let hc = HopfColeTransform::new(1.0);
        let l = DMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let z = DVector::from_vec(vec![1.0, 1.0]);
        assert!(!hc.verify_eigenfunction(&l, &z, 2.0, 1e-10));
    }

    #[test]
    fn test_temperature_scaling() {
        let hc_low = HopfColeTransform::new(0.1);
        let hc_high = HopfColeTransform::new(10.0);
        let v = DVector::from_vec(vec![0.0, 1.0]);
        let z_low = hc_low.value_to_desirability(&v);
        let z_high = hc_high.value_to_desirability(&v);
        // Higher temperature → flatter desirability
        assert!(z_low[1] < z_high[1]);
    }

    #[test]
    fn test_advantage() {
        let hc = HopfColeTransform::new(1.0);
        let p1 = DMatrix::identity(2, 2);
        let p2 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let passive = DMatrix::from_element(2, 2, 0.5);
        let z = DVector::from_vec(vec![1.0, 2.0]);
        let adv = hc.advantage(&[p1, p2], &passive, &z);
        assert_eq!(adv.len(), 2);
        // Advantage should be finite for valid z
        for a in &adv {
            for i in 0..2 {
                assert!(a[i].is_finite());
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_temperature() {
        HopfColeTransform::new(0.0);
    }
}
