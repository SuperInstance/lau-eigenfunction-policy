# lau-eigenfunction-policy

**Optimal RL policy as Dirichlet eigenfunction — solve RL via eigenvalues, not Bellman iteration.**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## What This Does

This crate implements a fundamentally different approach to solving Markov Decision Processes: instead of iteratively applying the Bellman equation, it **linearizes the Hamilton-Jacobi-Bellman (HJB) equation** via the Hopf-Cole transform and solves a **linear eigenvalue problem**.

The result: the optimal policy is the **principal eigenfunction** of the desirability-weighted transition operator, and the policy is extracted via the **Doob h-transform** (ground state transform of the heat flow).

The crate provides:
- **Hopf-Cole transform** — converts nonlinear HJB to linear eigenproblem
- **Dirichlet Laplacian** — transition operator with absorbing boundary conditions
- **Power iteration** — principal eigenvalue/eigenvector computation
- **Doob h-transform** — extracts the optimal policy from the ground state
- **Eigen-policy** — complete policy extraction with entropy, advantage, and evaluation
- **Spectral analysis** — spectral gap = convergence rate, mixing time analysis
- **WKB/semiclassical limit** — Varadhan's lemma connects value to graph distance

## Key Idea

```
HJB:  λV = c + λ log(P · e^{-V/λ})     ← nonlinear

Hopf-Cole:  z = e^{-V/λ}                 ← change of variables

Eigenproblem:  Lz = μz                   ← LINEAR!

where L = P · diag(e^{-c/λ})
```

The nonlinear Bellman equation, under the desirability transform `z = exp(-V/λ)`, becomes a **linear eigenvalue problem**. The principal eigenfunction `z*` is the desirability function, and the optimal policy is `π(a|s) ∝ (P_a z*)(s)`.

This is Emergent Theorem A: the optimal RL policy IS the ground state eigenfunction of the Dirichlet-form semigroup.

## Install

```toml
[dependencies]
lau-eigenfunction-policy = "0.1"
```

```bash
cargo add lau-eigenfunction-policy
```

Dependencies: `nalgebra` 0.33 (with serde), `serde` 1, `approx` 0.5 (dev).

## Quick Start

### Solve an MDP via Eigenfunctions

```rust
use lau_eigenfunction_policy::linear_mdp::LinearlySolvableMDP;

fn main() {
    // Create a 5-state chain MDP with goal at the end
    let mdp = LinearlySolvableMDP::chain(5, 0.5); // 5 states, temperature λ=0.5

    // Solve: find principal eigenfunction
    let (eigenvalue, desirability, value) = mdp.solve(1000, 1e-12);

    println!("Principal eigenvalue: {:.6}", eigenvalue);
    println!("Desirability z: {:?}", desirability.as_slice());
    println!("Value V = -λ log(z): {:?}", value.as_slice());
}
```

### Extract the Optimal Policy

```rust
use lau_eigenfunction_policy::{linear_mdp::LinearlySolvableMDP, policy::EigenPolicy};

let mdp = LinearlySolvableMDP::chain(5, 0.5);
let (eigenvalue, z, _v) = mdp.solve(1000, 1e-12);

// Extract policy from action transition kernels
let action_kernels: Vec<_> = (0..mdp.n_actions)
    .map(|a| mdp.action_operator(a))
    .collect();
let policy = EigenPolicy::from_eigenfunction(eigenvalue, z, &action_kernels, 0.5);

println!("Greedy policy: {:?}", policy.greedy_policy_vec());
println!("Entropy: {:?}", policy.entropy().as_slice());
```

### Grid World

```rust
use lau_eigenfunction_policy::linear_mdp::LinearlySolvableMDP;

let mdp = LinearlySolvableMDP::gridworld(4, 15, 1.0); // 4×4 grid, goal at state 15
let (eigenvalue, z, v) = mdp.solve(1000, 1e-12);
```

### Spectral Analysis

```rust
use lau_eigenfunction_policy::spectral::SpectralAnalysis;

let operator = mdp.dirichlet_operator();
let analysis = SpectralAnalysis::analyze(&operator, 1000, 1e-12);

println!("Spectral gap: {:.6}", analysis.spectral_gap());
println!("Mixing time: {:.2}", analysis.mixing_time());
println!("Convergence rate: {:.6}", analysis.convergence_rate());
```

## API Reference

### Modules

| Module | Description |
|--------|-------------|
| `hopf_cole` | Hopf-Cole transform: value ↔ desirability, linearization |
| `dirichlet` | Dirichlet Laplacian with absorbing boundary conditions |
| `eigen` | Power iteration, inverse iteration, deflation |
| `doob` | Doob h-transform: conditioned process = optimal policy |
| `linear_mdp` | Linearly-solvable MDP definition, grid/chain constructors |
| `policy` | Eigen-policy: action probabilities, entropy, advantage |
| `spectral` | Spectral gap, mixing time, convergence analysis |
| `wkb` | WKB/semiclassical limit, Varadhan distance |

### Core Types

| Type | Description |
|------|-------------|
| `HopfColeTransform` | Forward/inverse transform, linearization, advantage computation |
| `DirichletLaplacian` | Transition operator with Dirichlet BCs, Green's function, committor |
| `PowerIteration` | Eigenvalue solver with convergence rate estimation |
| `DoobHTransform` | Conditions passive dynamics into optimal policy |
| `LinearlySolvableMDP` | Full MDP with solve(), gridworld(), chain() constructors |
| `EigenPolicy` | Policy from eigenfunction: probabilities, greedy, entropy, KL |
| `SpectralAnalysis` | Full spectral decomposition, gap, mixing time |
| `WKBLimit` | Semiclassical limit, graph distances, Varadhan verification |

## How It Works

### Step 1: Linearize via Hopf-Cole

The nonlinear Bellman equation for a linearly-solvable MDP:

```
λV(s) = c(s) + λ log(Σ_{s'} P(s'|s) exp(-V(s')/λ))
```

Under `z(s) = exp(-V(s)/λ)`:

```
z = Lz / μ    where L = P · diag(exp(-c/λ))
```

This is a **linear eigenproblem**.

### Step 2: Find the Ground State

Power iteration on the Dirichlet operator `L` (with zero boundary at terminal states) finds the principal eigenfunction `z*` — the ground state desirability.

### Step 3: Extract Policy via Doob h-Transform

The optimal policy biases transitions toward high-desirability states:

```
π(a|s) = (P_a z*)(s) / Σ_{a'} (P_{a'} z*)(s)
```

This is exactly the **Doob h-transform** of the passive dynamics, conditioned on reaching the goal.

### Step 4: Verify via Spectral Gap

The spectral gap `λ₁ - λ₂` of the Dirichlet operator determines:
- **Convergence rate** of policy gradient
- **Mixing time** of the optimal policy
- **Effective dimension** of the policy space

### Step 5: WKB/Varadhan Verification

In the low-temperature limit (λ → 0), Varadhan's lemma guarantees:

```
-λ log(K_t(x,y)) → d(x,y)²
```

The value function converges to `V(x) ≈ ½ d(x, goal)²`, connecting RL to Riemannian geometry.

## The Math

### Linearly-Solvable MDPs

An MDP where the controlled transition kernel factors as:

```
P^u(s'|s) = P(s'|s) · exp(-c(s)/λ)
```

for passive dynamics P and cost c. This structure makes the HJB equation linearizable.

### Dirichlet Form

The operator `L = P · diag(e^{-c/λ})` defines a **Dirichlet form**:

```
E(f,f) = ⟨f, (I - L)f⟩
```

The principal eigenfunction minimizes the Rayleigh quotient `⟨f, Lf⟩/⟨f,f⟩`.

### Perron-Frobenius

Since L is a non-negative matrix, the Perron-Frobenius theorem guarantees:
- The principal eigenvalue is real and positive
- The principal eigenvector is non-negative (desirability ≥ 0)
- The spectral gap is positive for irreducible chains

### WKB Approximation

The semiclassical (λ → 0) limit gives:

```
z(x) ≈ exp(-d(x, goal)²/(2λ))
V(x) ≈ d(x, goal)²/2
```

This is the WKB approximation from quantum mechanics, adapted to the RL setting. Temperature `λ` plays the role of Planck's constant `ℏ`.

## Test Coverage

97 tests across all 8 modules:
- Hopf-Cole: forward/inverse transforms, roundtrips, linearization, advantage
- Dirichlet Laplacian: construction, interior extraction, eigenvalue, Green's function, committor, Rayleigh quotient
- Power iteration: identity, diagonal, dominant eigenvalue, convergence ratio, deflation, inverse iteration, residual norms
- Doob h-transform: stochasticity verification, biasing, Feynman-Kac, uniformity preservation
- Linearly-solvable MDP: creation, passive dynamics, weights, gridworld, chain, full solve
- Eigen-policy: greedy actions, value function, entropy, KL divergence, advantage, evaluation
- Spectral analysis: gap, mixing time, effective dimension, participation ratio, projection
- WKB limit: graph distances, Varadhan distance, semiclassical error, classical policy

## License

MIT
