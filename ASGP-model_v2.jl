# Computes the bipartite entanglement between the two bosonic modes
# AND
# Optimizes for minimum variance of
# (i)   standard quadratures (x & p types)
# (iii) mixed quadratures ("polaritonic")
# for the quantum battery model in arXiv:2409.08627

"""
We use the Meystre-Sargent convention while defining the quadratures.
In this convention [x, p] = i/2
    and the variance in the shot-noise limit is 0.25.
---
An alternative convention is to normalize the linear combination of operators by sqrt{2}.
In this convention [x, p] = i
    and the variance in the shot-noise limit is 0.5.
---
More details: arXiv: 1110.3234
"""

# =============================================================================
#  IMPORTS
# =============================================================================

using QuantumOptics
using LinearAlgebra
using Optim
using PyPlot
using Printf
using Random
using ProgressMeter
#using Base.Threads


# =============================================================================
#  PARAMETERS
# =============================================================================

# --- Fock-space truncation ---
const NA = 40    # dimension of A-boson Fock space
const NB = 40    # dimension of B-boson Fock space

# --- Model parameters ---
# Note: Only for n ≥ ωc/ωb is the energy of the initial charger state fully
#       transferred to the battery.
const n      = 4      # nonlinearity order: 1 quantum of 'a' ↔ n quanta of 'b'
const omegaB = 1.0
const omegaA = n * omegaB   # resonance condition; coupling commutes with quadratic terms
const g      = 1.0

# --- Initial-state parameters ---
const na_init = 16
const nb_init = 4
const Q       = n * na_init + nb_init   # conserved U(1) charge

# --- Time grid ---
const t_range = collect(0.0:0.0005:0.02)

# --- Optimizer settings ---
const OPT_ITERS   = 800    # max Nelder-Mead iterations per time step
const OPT_NSEEDS  = 12     # number of random seeds for multi-start (in addition to warm-start)
const OPT_RNG     = MersenneTwister(42)   # reproducible random restarts

# --- Output paths ---
mkpath(joinpath(@__DIR__, "data"))
mkpath(joinpath(@__DIR__, "plots"))
const FILENAME = "AGSP_omegaA($omegaA)_omegaB($omegaB)_g($g)_NA($NA)_NB($NB)_n($n)_naInit($na_init)_nbInit($nb_init)_tMax($(last(t_range)))_coherent_Kac"
const DATA_FILE = joinpath(@__DIR__, "data/$FILENAME.txt")


# =============================================================================
#  FUNCTIONS
# =============================================================================

"""
    iszero_op(op; tol) → Bool

Returns `true` when the Frobenius norm of `op` is below `tol`.
"""
function iszero_op(op::AbstractOperator; tol::Real = 1e-10)
    return norm(op.data) <= tol
end


"""
    commute(A, B; tol) → Bool

Returns `true` when [A, B] ≈ 0.
"""
function commute(A::AbstractOperator, B::AbstractOperator; tol::Real = 1e-10)
    return iszero_op(A * B - B * A; tol = tol)
end


"""
    chop_small(x, tol)

Zeroes out real / imaginary parts whose magnitude is below `tol`.
"""
function chop_small(x, tol)
    if x isa Complex
        re = abs(real(x)) < tol ? 0.0 : real(x)
        im = abs(imag(x)) < tol ? 0.0 : imag(x)
        return complex(re, im)
    else
        return abs(x) < tol ? 0.0 : x
    end
end


"""
    variance_op(X, ψ_series) → Vector{Float64}

Returns Var[X] = ⟨X²⟩ − ⟨X⟩² for each state in `ψ_series`.
`real` guards against tiny numerical imaginary parts.
"""
function variance_op(X::AbstractOperator, ψ_series::AbstractVector{<:Ket})
    μ  = expect(X,     ψ_series)
    μ2 = expect(X * X, ψ_series)
    return real.(μ2 .- μ.^2)
end


"""
    vonNeumann_entropy(ρ; base, tol) → Float64

Von Neumann entropy S(ρ) = −Tr[ρ log_b ρ].
Tiny / negative eigenvalues (numerical noise) are clipped to zero.
"""
function vonNeumann_entropy(ρ::AbstractOperator; base::Real = 2, tol::Real = 1e-12)
    vals    = real.(eigvals(Matrix(ρ.data)))
    vals    = map(x -> x < tol ? 0.0 : x, vals)
    logbase = log(base)
    S       = sum(p > tol ? -p * log(p) / logbase : 0.0 for p in vals)
    return S
end


"""
    X_op(θ, η, ϕ, a, b) → AbstractOperator

Polaritonic quadrature X(θ,η,ϕ) = (c + c†)/2
with c = e^{iϕ}(a cosθ + e^{iη} b sinθ).
"""
function X_op(θ::Real, η::Real, ϕ::Real,
              a::AbstractOperator, b::AbstractOperator)
    c = exp(1im * ϕ) * (a * cos(θ) + exp(1im * η) * b * sin(θ))
    return (c + dagger(c)) / 2
end


"""
    varX(params, ψ, a, b) → Float64

Objective function: Var[X(θ,η,ϕ)] for a single state ψ.
Parameters are wrapped into their principal domains so the optimizer
can run unconstrained.
"""
function varX(params::AbstractVector{<:Real},
              ψ::Ket,
              a::AbstractOperator, b::AbstractOperator)
    θ = mod(params[1], π/2)   # mixing angle ∈ [0, π/2)
    η = mod(params[2], 2π)    # relative phase ∈ [0, 2π)
    ϕ = mod(params[3], 2π)    # overall phase  ∈ [0, 2π)
    X  = X_op(θ, η, ϕ, a, b)
    μ  = expect(X, ψ)
    μ2 = expect(X * X, ψ)
    return real(μ2 - μ^2)
end


"""
    optimize_varX_series(psi_t, t_range, a, b; iters, n_seeds, rng)
                        → (min_varX_t, θ_opt_t, η_opt_t, ϕ_opt_t)

Minimises Var[X(θ,η,ϕ)] at each time step using a **multi-start Nelder-Mead**
strategy:

  1. `n_seeds` candidates are drawn uniformly from the parameter box
     [0, π/2] x [0, 2π] x [0, 2π].
  2. The warm-start from the previous step's optimum is added as one
     further candidate (good for slowly-varying landscapes).
  3. All candidates are minimised; the global minimum is kept.

This guards against NelderMead getting trapped in a local minimum — a real
risk for the 2π-periodic parameters η and ϕ.
"""
function optimize_varX_series(
        psi_t, t_range, a, b;
        iters::Int   = OPT_ITERS,
        n_seeds::Int = OPT_NSEEDS,
        rng          = OPT_RNG)

    N = length(psi_t)
    min_varX_t = zeros(N)
    θ_opt_t    = zeros(N)
    η_opt_t    = zeros(N)
    ϕ_opt_t    = zeros(N)

    warm_start  = [π/2, 0.0, 0.0]   # initial guess for t=0
    opts        = Optim.Options(iterations = iters, store_trace = false, show_trace = false)

    prog = Progress(N; desc="Optimising quadrature: ", barlen=40, showspeed=true,output=stderr)

    for (i, ψ) in enumerate(psi_t)
        #println("Optimizing for t = ", t_range[i])

        obj(p) = varX(p, ψ, a, b)

        # --- Build candidate starting points ---
        # Random seeds: θ ∈ [0, π/2], η,ϕ ∈ [0, 2π]
        seeds = [
            [rand(rng) * π/2, rand(rng) * 2π, rand(rng) * 2π]
            for _ in 1:n_seeds
        ]
        push!(seeds, warm_start)   # always include the warm-start

        # --- Run Nelder-Mead from every seed; keep global minimum ---
        best_val    = Inf
        best_params = warm_start

        for s in seeds
            res = Optim.optimize(obj, s, NelderMead(), opts)
            if Optim.minimum(res) < best_val
                best_val    = Optim.minimum(res)
                best_params = collect(Optim.minimizer(res))
            end
        end

        min_varX_t[i] = best_val
        θ_opt_t[i]    = mod(best_params[1], π/2)
        η_opt_t[i]    = mod(best_params[2], 2π)
        ϕ_opt_t[i]    = mod(best_params[3], 2π)

        # Warm-start next step from this step's optimum
        warm_start = best_params

        next!(prog; showvalues = [("t", @sprintf("%.4f", t_range[i]))])
    end

    return min_varX_t, θ_opt_t, η_opt_t, ϕ_opt_t
end


# =============================================================================
#  MAIN EXECUTION
# =============================================================================

# ── Step 1: Build Hilbert space ───────────────────────────────────────────────
println("Building Hilbert space...")
a_fock        = FockBasis(NA)
b_fock        = FockBasis(NB)
composite_fock = tensor(a_fock, b_fock)

# Fundamental operators
a  = embed(composite_fock, 1, destroy(a_fock))
at = embed(composite_fock, 1, create(a_fock))
na = embed(composite_fock, 1, number(a_fock))

b  = embed(composite_fock, 2, destroy(b_fock))
bt = embed(composite_fock, 2, create(b_fock))
nb = embed(composite_fock, 2, number(b_fock))


# ── Step 2: Build Hamiltonian ─────────────────────────────────────────────────
println("Building Hamiltonian...")
H0 = omegaB * na + omegaA * nb +
     (g / sqrt(factorial(n - 1))) * (a * bt^n + at * b^n)
H  = (H0 + dagger(H0)) / 2   # enforce numerical Hermiticity
println("Hamiltonian built.")


# ── Step 3: Prepare initial state ────────────────────────────────────────────
println("Preparing initial state...")

# --- Option A: Coherent state (active) ---
alpha   = sqrt(na_init) * exp(-1im * π/2)
beta    = sqrt(nb_init) * exp(1im  * 0.0)
psi0    = tensor(coherentstate(a_fock, alpha), coherentstate(b_fock, beta))

# --- Option B: Fock state (commented out) ---
# psi0 = tensor(fockstate(a_fock, na_init), fockstate(b_fock, nb_init))

# --- Option C: Squeezed state (commented out) ---
# ζa  = asinh(sqrt(na_init)) * exp(1im * 0.0)
# ζb  = asinh(sqrt(nb_init)) * exp(1im * 0.0)
# psi0 = tensor(squeeze(a_fock, ζa) * fockstate(a_fock, 0),
#               squeeze(b_fock, ζb) * fockstate(b_fock, 0))

# --- Check conserved charge ---
Qop = n * na + nb
@show real(expect(Qop, psi0))
@show sqrt(real(expect(Qop^2, psi0) - expect(Qop, psi0)^2))


# ── Step 4: Time-evolve ───────────────────────────────────────────────────────
println("Running time evolution...")
t_out, psi_t = timeevolution.schroedinger(t_range, psi0, H)
println("Time evolution complete.")


# ── Step 5: Compute standard observables ─────────────────────────────────────
println("Computing occupation numbers and quadrature variances...")
na_t = real(expect(na, psi_t))
nb_t = real(expect(nb, psi_t))

x_a = (a  + at)          / 2
x_b = (b  + bt)          / 2
p_a = (a  - at)          / (2im)
p_b = (b  - bt)          / (2im)

var_xa = variance_op(x_a, psi_t)
var_xb = variance_op(x_b, psi_t)
var_pa = variance_op(p_a, psi_t)
var_pb = variance_op(p_b, psi_t)


# ── Step 6: Compute entanglement entropy ─────────────────────────────────────
println("Computing entanglement entropy...")
S_ab_t = [vonNeumann_entropy(ptrace(dm(ψ), [2])) for ψ in psi_t]


# ── Step 7: Optimise polaritonic quadrature ───────────────────────────────────
println("Optimising polaritonic quadrature variance...")
min_varX_t, θ_opt_t, η_opt_t, ϕ_opt_t =
    optimize_varX_series(psi_t, t_range, a, b)
println("Optimisation complete.")


# ── Step 8: Save data ─────────────────────────────────────────────────────────
println("Writing data to ", DATA_FILE)
open(DATA_FILE, "w") do io
    println(io, "# t\tna\tnb\tvar_xa\tvar_xb\tvar_pa\tvar_pb\tS_ab\tmin_varX\ttheta_opt\teta_opt\tphi_opt")
    for i in eachindex(t_range)
        @printf(io,
            "%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\t%0.10f\n",
            t_range[i],
            na_t[i],    nb_t[i],
            var_xa[i],  var_xb[i],  var_pa[i],  var_pb[i],
            S_ab_t[i],
            min_varX_t[i],
            θ_opt_t[i], η_opt_t[i], ϕ_opt_t[i])
    end
end
println("Data saved.")


# ── Step 9: Plot standard quadrature observables ──────────────────────────────
fig = figure(figsize=(12, 6))

subplot(2, 3, 1)
plot(t_range, na_t)
axhline(na_init, linestyle="--", color="0.0")
axhline(nb_init, linestyle="--", color="0.5")
xlabel(L"t"); ylabel(L"\langle \hat{n}_a \rangle")

subplot(2, 3, 4)
plot(t_range, nb_t)
axhline(na_init, linestyle="--", color="0.0")
axhline(nb_init, linestyle="--", color="0.5")
xlabel(L"t"); ylabel(L"\langle \hat{n}_b \rangle")

subplot(2, 3, 2)
plot(t_range, var_xa)
axhline(0.25, linestyle="--", color="0.0")
xlabel(L"t"); ylabel(L"\langle (\Delta \hat{x}_a)^2 \rangle")

subplot(2, 3, 5)
plot(t_range, var_xb)
axhline(0.25, linestyle="--", color="0.0")
xlabel(L"t"); ylabel(L"\langle (\Delta \hat{x}_b)^2 \rangle")

subplot(2, 3, 3)
plot(t_range, var_pa)
axhline(0.25, linestyle="--", color="0.0")
xlabel(L"t"); ylabel(L"\langle (\Delta \hat{p}_a)^2 \rangle")

subplot(2, 3, 6)
plot(t_range, var_pb)
axhline(0.25, linestyle="--", color="0.0")
xlabel(L"t"); ylabel(L"\langle (\Delta \hat{p}_b)^2 \rangle")

tight_layout()
fig.savefig(joinpath(@__DIR__, "plots/na_nb_vs_t_$FILENAME.pdf"))
display(fig); show()


# ── Step 10: Plot optimised polaritonic quadrature ────────────────────────────
fig_opt = figure(figsize=(10, 6))

subplot(2, 2, 1)
plot(t_range, min_varX_t, marker="o", linestyle="none")
axhline(0.25, linestyle="--", color="0.0")
ylim([0, 0.3])
xlabel(L"t")
ylabel(L"\min_{\theta,\eta,\phi}\,\langle (\Delta \hat{X})^2 \rangle (t)")

subplot(2, 2, 2)
plot(t_range, θ_opt_t ./ π, marker="o", linestyle="none")
xlabel(L"t"); ylabel(L"\theta_\mathrm{opt}/\pi")

subplot(2, 2, 3)
plot(t_range, η_opt_t ./ π, marker="o", linestyle="none")
xlabel(L"t"); ylabel(L"\eta_\mathrm{opt}/\pi")

subplot(2, 2, 4)
plot(t_range, ϕ_opt_t ./ π, marker="o", linestyle="none")
xlabel(L"t"); ylabel(L"\phi_\mathrm{opt}/\pi")

tight_layout()
fig_opt.savefig(joinpath(@__DIR__, "plots/optX_vs_t_$FILENAME.pdf"))
display(fig_opt); show()


# ── (Optional) Plot entanglement entropy ─────────────────────────────────────
# fig_ent = figure(figsize=(6, 4))
# plot(t_range, S_ab_t)
# xlabel(L"t"); ylabel(L"S_{a:b}(t)")
# tight_layout()
# fig_ent.savefig(joinpath(@__DIR__, "plots/entropy_vs_t_$FILENAME.pdf"))
# display(fig_ent); show()
