"""
Sakurai & Napolitano, Modern Quantum Mechanics (3rd ed.)
Problem 2.22 -- Animation of an arbitrary superposition of SHO stationary states

PHYSICS SUMMARY (every equation number below is from the primary text)
------------------------------------------------------------------------
We are given an initial state

        |alpha> = sum_n c_n |n>,          sum_n |c_n|^2 = 1

and asked to animate Psi(x,t) = <x|alpha,t0=0;t> and rho(x,t) = |Psi(x,t)|^2.

STEP 1 -- Time evolution of the coefficients (purely algebraic, no PDE solve)
    Because H|n> = (n+1/2) hbar*omega |n>                         Eq. (2.129)
    and      U(t) = exp(-i H t / hbar)                            Eq. (2.28)
    we get immediately
        c_n(t) = c_n(0) * exp[-i (n+1/2) omega t].
    This is the entire "dynamics" -- each number-basis coefficient just
    picks up a phase at its own Bohr frequency. No numerical time-stepping
    of a differential equation is needed.

STEP 2 -- The spatial eigenfunctions
    u_n(x) = c_n H_n( x*sqrt(m*omega/hbar) ) * exp(-m*omega*x^2 / 2*hbar)   Eq. (2.232)
    with the normalization constant c_n fixed by the orthogonality relation
    (2.233), derived explicitly in Problem 2.25, and tabulated compactly in
    Appendix (B.27):
        u_n(xi) = (2^n n!)^(-1/2) (m*omega/pi/hbar)^(1/4) exp(-xi^2/2) H_n(xi)
    with the dimensionless coordinate  xi = x*sqrt(m*omega/hbar) = x/x0,
    x0 = sqrt(hbar/m/omega)                                         Eq. (2.150)

    NUMERICAL NOTE: for n larger than ~60, H_n(xi) and exp(-xi^2/2) individually
    overflow/underflow in floating point even though their product u_n(xi) is a
    perfectly well-behaved O(1) function. We therefore evolve the *normalized
    functions themselves* via a stable three-term recurrence (mathematically
    equivalent to (2.232)-(2.233), just algebraically repackaged to avoid
    forming the dangerous large/small factors separately):

        psi_0(xi) = pi^(-1/4) exp(-xi^2/2)
        psi_1(xi) = sqrt(2) * xi * psi_0(xi)
        psi_n(xi) = sqrt(2/n) * xi * psi_{n-1}(xi) - sqrt((n-1)/n) * psi_{n-2}(xi)

STEP 3 -- Assemble and superpose
    Psi(x,t) = sum_n c_n(t) * u_n(x)
    rho(x,t) = |Psi(x,t)|^2

CHECKS REQUIRED BY THE PROBLEM STATEMENT
    (A) Pure eigenstate (c_n = delta_{n,n0}): rho(x,t) must be time-independent.
        This is a direct animation check of Eq. (2.45): <B> for an energy
        eigenstate does not depend on t (a "stationary state").
    (B) Classical-looking combination: the equal-weight two-level state of
        Problem 2.19, |alpha> = (|0> + |1>)/sqrt(2), should show
        <x>(t) proportional to cos(omega t)  (Problem 2.19(b) result),
        while the packet visibly changes width ("breathes") as it moves.
    (C) Coherent state |lambda> (Problem 2.21): for |lambda|^2 >> 1 it should
        oscillate RIGIDLY (constant width sqrt(hbar/2 m omega), Problem 2.21(b))
        with amplitude sqrt(2 hbar |lambda|^2 / m omega) -- the closest quantum
        mechanics comes to classical motion. The coefficients are the Poisson
        weights of Problem 2.21(c):
            f(n) = exp(-|lambda|^2/2) * lambda^n / sqrt(n!)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import factorial
from pathlib import Path


# ----------------------------------------------------------------------
# STEP 2 (code): stable, normalized SHO eigenfunctions psi_n(xi)
# ----------------------------------------------------------------------
def sho_eigenfunctions(xi, n_max):
    """
    Return an array of shape (n_max+1, len(xi)) whose n-th row is the
    normalized dimensionless SHO eigenfunction psi_n(xi), equivalent to
    Eq. (2.232) with the normalization constant of Problem 2.25 / Eq. (2.233),
    but computed via the numerically stable recurrence described above.

    xi    : dimensionless position array, xi = x / x0, x0 = sqrt(hbar/m/omega)  [Eq. (2.150)]
    n_max : highest quantum number needed
    """
    xi = np.asarray(xi)
    psi = np.zeros((n_max + 1, xi.size))

    # psi_0(xi) = pi^(-1/4) exp(-xi^2/2)          -- ground state, Eq. (2.151)
    psi[0] = np.pi ** (-0.25) * np.exp(-0.5 * xi ** 2)

    if n_max >= 1:
        # psi_1(xi) = sqrt(2) * xi * psi_0(xi)     -- first excited state
        psi[1] = np.sqrt(2.0) * xi * psi[0]

    # Three-term recurrence for n >= 2 (stable: no huge/tiny numbers formed)
    for n in range(2, n_max + 1):
        psi[n] = np.sqrt(2.0 / n) * xi * psi[n - 1] - np.sqrt((n - 1.0) / n) * psi[n - 2]

    return psi


# ----------------------------------------------------------------------
# STEP 1 + STEP 3 (code): build Psi(x,t) from coefficients c_n
# ----------------------------------------------------------------------
def evolve_wavefunction(c_n, x, t_array, m=1.0, hbar=1.0, omega=1.0):
    """
    c_n     : 1D complex array of initial coefficients c_n(0), c_n[n] = <n|alpha>
    x       : 1D array of position grid points (physical units, not dimensionless)
    t_array : 1D array of times at which to evaluate Psi(x,t)

    Returns Psi of shape (len(t_array), len(x)), complex.
    """
    n_max = len(c_n) - 1
    x0 = np.sqrt(hbar / (m * omega))            # length scale, Eq. (2.150)
    xi = x / x0

    # Dimensionless-to-physical normalization: u_n(x) dx = psi_n(xi) dxi,
    # so u_n(x) = psi_n(xi) / sqrt(x0)  to keep integral( |u_n(x)|^2 dx ) = 1.
    psi_n_of_x = sho_eigenfunctions(xi, n_max) / np.sqrt(x0)   # shape (n_max+1, len(x))

    # Bohr frequencies E_n/hbar = (n + 1/2) omega, from Eq. (2.129)/(2.130)
    n_vals = np.arange(n_max + 1)
    E_n = (n_vals + 0.5) * hbar * omega

    Psi = np.zeros((len(t_array), len(x)), dtype=complex)
    for it, t in enumerate(t_array):
        # STEP 1: c_n(t) = c_n(0) exp(-i E_n t / hbar)      [Eq. (2.28) acting via Eq. (2.129)]
        c_n_t = c_n * np.exp(-1j * E_n * t / hbar)
        # STEP 3: Psi(x,t) = sum_n c_n(t) u_n(x)
        Psi[it] = c_n_t @ psi_n_of_x

    return Psi


# ----------------------------------------------------------------------
# Convenience: coefficients for special cases used in the required checks
# ----------------------------------------------------------------------
def eigenstate_coeffs(n0, n_max):
    """Check (A): a single pure eigenstate |n0>."""
    c = np.zeros(n_max + 1, dtype=complex)
    c[n0] = 1.0
    return c


def classical_like_coeffs(n_max):
    """Check (B): Problem 2.19's state, (|0> + |1>)/sqrt(2)."""
    c = np.zeros(n_max + 1, dtype=complex)
    c[0] = 1.0 / np.sqrt(2.0)
    c[1] = 1.0 / np.sqrt(2.0)
    return c


def coherent_state_coeffs(lam, n_max):
    """
    Check (C): coherent state |lambda>, Problem 2.21.
    f(n) = exp(-|lambda|^2/2) * lambda^n / sqrt(n!)          [Problem 2.21(c)]
    n_max should be chosen well above |lambda|^2 (the Poisson mean) so the
    tail that is cut off carries negligible probability.
    """
    n_vals = np.arange(n_max + 1)
    # Direct formula (n_max here is modest -- a few tens -- so factorial() is fine;
    # for very large n_max one would move to a log-space / Stirling evaluation instead).
    f = np.array(
        [np.exp(-0.5 * abs(lam) ** 2) * (lam ** n) / np.sqrt(float(factorial(n)))
         for n in n_vals],
        dtype=complex,
    )
    return f


# ----------------------------------------------------------------------
# Animation driver
# ----------------------------------------------------------------------
def animate_state(c_n, label, m=1.0, hbar=1.0, omega=1.0,
                   x_range=8.0, n_x=600, n_frames=120, n_periods=2,
                   save_path=None):
    """
    Build and (optionally) save an animation of Re[Psi(x,t)] and rho(x,t)=|Psi|^2
    side by side, over n_periods classical periods T = 2*pi/omega.
    """
    c_n = c_n / np.sqrt(np.sum(np.abs(c_n) ** 2))   # enforce normalization, sum|c_n|^2=1
    x = np.linspace(-x_range, x_range, n_x)
    T = 2 * np.pi / omega
    t_array = np.linspace(0.0, n_periods * T, n_frames)

    Psi = evolve_wavefunction(c_n, x, t_array, m=m, hbar=hbar, omega=omega)
    rho = np.abs(Psi) ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(label)

    # Panel 1: Re[Psi(x,t)]  (the wave function itself, as requested)
    line1, = ax1.plot(x, Psi[0].real, lw=2)
    ax1.set_ylim(1.2 * Psi.real.min(), 1.2 * Psi.real.max())
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"Re $\Psi(x,t)$")
    ax1.set_title("Wave function")

    # Panel 2: rho(x,t) = |Psi(x,t)|^2 (the probability density, as requested)
    line2, = ax2.plot(x, rho[0], lw=2, color="crimson")
    ax2.set_ylim(0, 1.2 * rho.max())
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$|\Psi(x,t)|^2$")
    ax2.set_title("Probability density")

    time_text = ax1.text(0.02, 0.92, "", transform=ax1.transAxes)

    def update(frame):
        line1.set_ydata(Psi[frame].real)
        line2.set_ydata(rho[frame])
        time_text.set_text(f"t = {t_array[frame]:.2f} (t/T = {t_array[frame]/T:.2f})")
        return line1, line2, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=40, blit=True
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow", fps=25)
        print(f"Saved animation to {save_path}")

    return anim, x, t_array, Psi


# ----------------------------------------------------------------------
# Numerical verification of the three required checks (no display needed)
# ----------------------------------------------------------------------
def verify_checks(m=1.0, hbar=1.0, omega=1.0, eigenstate_n=3):
    x = np.linspace(-10, 10, 2000)

    # ---- Check (A): pure eigenstate -> rho(x,t) must be time-independent ----
    n0 = eigenstate_n
    c = eigenstate_coeffs(n0, n_max=n0)
    t_array = np.linspace(0, 4 * np.pi / omega, 15)
    Psi = evolve_wavefunction(c, x, t_array, m, hbar, omega)
    rho = np.abs(Psi) ** 2
    spread = np.max(np.abs(rho - rho[0]))
    print(f"[Check A] max_t |rho(x,t) - rho(x,0)| for pure |n={n0}> "
          f"= {spread:.3e}  (should be ~1e-10, i.e. stationary, cf. Eq. 2.45)")

    # ---- Check (B): classical-like combo -> <x>(t) ~ cos(omega t) ----
    n_max = 5
    c = classical_like_coeffs(n_max)
    t_array = np.linspace(0, 2 * 2 * np.pi / omega, 60)
    Psi = evolve_wavefunction(c, x, t_array, m, hbar, omega)
    rho = np.abs(Psi) ** 2
    dx = x[1] - x[0]
    x_expect = np.sum(rho * x[None, :], axis=1) * dx
    # analytic prediction from Problem 2.19(b): <x>(t) = sqrt(hbar/2 m omega) * cos(omega t)
    x_expect_analytic = np.sqrt(hbar / (2 * m * omega)) * np.cos(omega * t_array)
    err = np.max(np.abs(x_expect - x_expect_analytic))
    print(f"[Check B] max |<x>(t)_numeric - <x>(t)_analytic| for (|0>+|1>)/sqrt(2) "
          f"= {err:.3e}  (should be small; confirms Problem 2.19(b))")

    # ---- Check (C): coherent state -> rigid oscillation, constant width ----
    lam = 3.0 + 0.0j          # |lambda|^2 = 9, a few quanta -> already fairly classical
    n_max = 40                # >> |lambda|^2, negligible truncated tail
    c = coherent_state_coeffs(lam, n_max)
    t_array = np.linspace(0, 2 * np.pi / omega, 40)
    Psi = evolve_wavefunction(c, x, t_array, m, hbar, omega)
    rho = np.abs(Psi) ** 2
    x_expect = np.sum(rho * x[None, :], axis=1) * dx
    x2_expect = np.sum(rho * (x ** 2)[None, :], axis=1) * dx
    var_x = x2_expect - x_expect ** 2
    print(f"[Check C] coherent state width Var(x)(t): min={var_x.min():.5f}, "
          f"max={var_x.max():.5f}  (should both equal hbar/2 m omega = "
          f"{hbar/(2*m*omega):.5f}, i.e. constant width -> rigid motion, Problem 2.21b)")


if __name__ == "__main__":
    eigenstate_n = 4

    # Run the three required numerical checks
    verify_checks(eigenstate_n=eigenstate_n)

    # Build the three requested animations and save them as GIFs
    animate_state(
        eigenstate_coeffs(eigenstate_n, eigenstate_n),
        label=f"Check (A): pure eigenstate |n={eigenstate_n}> (should look frozen)",
        n_periods=1,
        save_path=f"./Chapter2/images/check_A_eigenstate_n{eigenstate_n}.gif",
    )

    animate_state(
        classical_like_coeffs(5),
        label="Check (B): (|0>+|1>)/sqrt(2), Problem 2.19 -- classical-like oscillation",
        n_periods=2,
        save_path="./Chapter2/images/check_B_classical_combo.gif",
    )

    animate_state(
        coherent_state_coeffs(3.0 + 0.0j, 40),
        label="Check (C): coherent state |lambda=3>, Problem 2.21 -- rigid oscillation",
        n_periods=2,
        save_path="./Chapter2/images/check_C_coherent_state.gif",
    )
