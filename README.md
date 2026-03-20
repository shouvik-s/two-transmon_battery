# ASGP-model v2 — Squeezing & Entanglement in a Nonlinear Quantum Battery

Simulation code for the quantum battery model described in
[arXiv:2409.08627](https://arxiv.org/abs/2409.08627).

The script computes:
1. **Bipartite entanglement entropy** between two bosonic modes (a, b).
2. **Standard quadrature variances** — $\langle(\Delta\hat{x}_{a,b})^2\rangle$ and $\langle(\Delta\hat{p}_{a,b})^2\rangle$.
3. **Optimised polaritonic (mixed) quadrature variance** — minimised over the mixing angle $\theta$, relative phase $\eta$, and overall phase $\phi$ at each time step.

---

## Physical Model

The Hamiltonian is

$$H = \omega_b\,\hat{n}_a + \omega_a\,\hat{n}_b + \frac{g}{\sqrt{(n-1)!}}\!\left(\hat{a}\,\hat{b}^{\dagger n} + \hat{a}^\dagger\hat{b}^n\right)$$

where $\omega_a = n\,\omega_b$ (resonance condition).
The coupling conserves the U(1) charge $Q = n\,\hat{n}_a + \hat{n}_b$.

### Quadrature convention (Meystre–Sargent)

$$\hat{x} = \frac{\hat{a}+\hat{a}^\dagger}{2}, \quad \hat{p} = \frac{\hat{a}-\hat{a}^\dagger}{2i}, \quad [\hat{x},\hat{p}] = \frac{i}{2}$$

Shot-noise limit: $\langle(\Delta\hat{x})^2\rangle = 0.25$.
See [arXiv:1110.3234](https://arxiv.org/abs/1110.3234) for a comparison of conventions.

### Polaritonic quadrature

$$\hat{X}(\theta,\eta,\phi) = \frac{\hat{c}+\hat{c}^\dagger}{2}, \quad \hat{c} = e^{i\phi}\!\left(\hat{a}\cos\theta + e^{i\eta}\hat{b}\sin\theta\right)$$

The minimum variance over $(\theta,\eta,\phi)$ quantifies two-mode squeezing beyond the individual-mode level.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NA`, `NB` | 40 | Fock-space truncation dimensions |
| `n` | 4 | Nonlinearity order ($n$ quanta of $b$ per quantum of $a$) |
| `omegaB` | 1.0 | Frequency of mode $b$ |
| `omegaA` | `n * omegaB` | Frequency of mode $a$ (resonance condition) |
| `g` | 1.0 | Coupling strength |
| `na_init` | 16 | Initial excitation of mode $a$ |
| `nb_init` | 4 | Initial excitation of mode $b$ |
| `t_range` | 0 to 0.02 (step 5×10⁻⁴) | Time grid |
| `OPT_ITERS` | 800 | Max Nelder-Mead iterations per time step |
| `OPT_NSEEDS` | 12 | Random restarts for the multi-start optimizer |

---

## Initial States

Three options are provided in the code (switch by commenting/uncommenting):

- **Coherent state** *(default)* — $|\alpha\rangle_a \otimes |\beta\rangle_b$
- **Fock state** — $|n_a\rangle \otimes |n_b\rangle$
- **Squeezed vacuum** — $\hat{S}(\zeta_a)|0\rangle_a \otimes \hat{S}(\zeta_b)|0\rangle_b$

---

## Workflow

```
1. Build composite Fock space (NA+1) × (NB+1)
2. Construct Hamiltonian H and enforce numerical Hermiticity
3. Prepare initial state ψ₀; verify conserved charge Q
4. Schrödinger time evolution → {ψ(t)}
5. Compute ⟨n_a⟩, ⟨n_b⟩, quadrature variances
6. Compute bipartite von Neumann entropy S_{a:b}(t)
7. Multi-start Nelder-Mead optimisation of Var[X(θ,η,ϕ)] at each t
8. Save all observables to data/<FILENAME>.txt
9. Plot standard quadrature observables → plots/na_nb_vs_t_<FILENAME>.pdf
10. Plot optimised polaritonic quadrature → plots/optX_vs_t_<FILENAME>.pdf
```

---

## Output

### Data file (`data/<FILENAME>.txt`)
Tab-separated, one row per time step:

| Column | Description |
|--------|-------------|
| `t` | Time |
| `na`, `nb` | Mean occupation numbers |
| `var_xa`, `var_xb` | Position quadrature variances |
| `var_pa`, `var_pb` | Momentum quadrature variances |
| `S_ab` | Von Neumann entanglement entropy (base 2) |
| `min_varX` | Minimised polaritonic quadrature variance |
| `theta_opt`, `eta_opt`, `phi_opt` | Optimal mixing angle and phases |

### Plots (`plots/`)
- `na_nb_vs_t_<FILENAME>.pdf` — 2×3 panel: $\langle\hat{n}_{a,b}\rangle$ and all four standard quadrature variances vs. time.
- `optX_vs_t_<FILENAME>.pdf` — 2×2 panel: $\min\langle(\Delta\hat{X})^2\rangle$ and the optimal angles $\theta,\eta,\phi$ vs. time.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [`QuantumOptics.jl`](https://github.com/qojulia/QuantumOptics.jl) | Fock spaces, operators, time evolution |
| `LinearAlgebra` | Eigenvalue decomposition (entropy) |
| [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) | Nelder-Mead optimisation |
| [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) | Matplotlib-based plotting |
| `Printf`, `Random` | Formatted output, reproducible RNG |
| [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl) | Progress bar during optimisation |

---

## Usage

```julia
julia ASGP-model_v2.jl
```

Output files are written to `data/` and `plots/` subdirectories (created automatically).

---

## Reference

> *Quantum battery model* — arXiv:2409.08627
> *Quadrature conventions* — arXiv:1110.3234
