# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Emmanuel Oguadimma
MTH 520
30/01/2025
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j, k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j, k))*20 - 10
    A[A[:, -1] < 0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k, :] @ x
    b[k:] = A[k:, :] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k, :].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems 1 --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """
    Solve the linear program min c^T x,  subject to  A x = b, x >= 0
    via an Interior‐Point method.
    -------------------------------------------------------------------
    Parameters:
        A      ((m,n) ndarray): equality constraint matrix (full row rank)
        b      ((m,) ndarray):  equality constraint vector
        c      ((n,) ndarray):  objective‐coefficient vector
        niter  (int)          :  maximum number of Newton iterations
        tol    (float)        :  stopping tolerance
        verbose (bool)        :  if True, print residual norms each iteration
    Returns:
        x      ((n,) ndarray):  the optimal primal solution
        val    (float)        :  the minimum objective value c^T x
    """

    m, n = A.shape[0], A.shape[1]

    # -----------------------------------------------------------------
    # (1)  Define F(x, λ, μ)  as a nested function, so that it can see A, b, c
    #      If x is length-n, λ is length-m, μ is length-n, then F returns
    #      [ A^T λ + μ - c;  A x - b;  x * μ ] which has size  n + m + n = 2n + m.
    # -----------------------------------------------------------------
    def F(x, lam, mu):
        """
        Vector‐valued function for the KKT‐residuals:
          r_dual   = A^T lam + mu - c        (length n)
          r_primal = A x - b                 (length m)
          r_cent   = x * mu                  (length n)

        Returns a single 1D array of length (2*n + m).
        """
        # dual residual:   A^T λ + μ - c     ∈ R^n
        r_dual   = A.T.dot(lam) + mu - c

        # primal residual: A x - b           ∈ R^m
        r_primal = A.dot(x) - b

        # complementarity: x ∘ μ             ∈ R^n
        r_cent   = x * mu

        # stack into one long 1D array of length (n + m + n):
        return np.concatenate([r_dual, r_primal, r_cent])

    # -----------------------------------------------------------------
    # (2)  Now you can call F(x, lam, mu) wherever you need within
    #      your interior‐point iterations to build the full residual.
    #
    #      For example, in each Newton step you might do:
    #
    #         resid = F(x, lam, mu)
    #         if norm(resid) < tol:  break
    #         … build Jacobian J_F, solve J_F * delta = -resid, …
    #
    # (The rest of your Newton/affine‐scaling logic goes here.)
    # -----------------------------------------------------------------

    #
    #  [ …  your existing IPM code (Newton‐system, line search, updates…)  … ]
    #

    # For the sake of completeness, suppose we default‐return the starting point:
    x0, lam0, mu0 = starting_point(A, b, c)

    # (In a real implementation you would iterate, using F, until convergence.)
    return x0, float(c.dot(x0))



# Problems 2
def compute_search_direction(A, b, c,
                             x,          # current primal vector, shape (n,)
                             lambda_vec, # current λ,     shape (m,)
                             mu_vec,     # current μ,     shape (n,)
                             sigma=0.1): # centering parameter (σ)
    """
    Solve:
        DF(x,λ,μ) · [Δx; Δλ; Δμ]  =  −F(x,λ,μ)  +  [0; 0; σ·τ·e],
    where
        F(x,λ,μ) = [ A^T λ + μ − c;  A x − b;  x ∘ μ ],
        DF = [   0      A^T     I ;
                A       0      0 ;
                M       0      X ],
        X = diag(x),     M = diag(mu),
        τ = (μ^T x)/n,    e = ones(n).
    -----------------------------------------------------------------------
    Inputs:
      A         : (m×n) ndarray of constraint coefficients
      b         : (m,)   ndarray of RHS
      c         : (n,)   ndarray of objective coefficients
      x         : (n,)   current primal iterate
      lambda_vec: (m,)   current dual‐equality multipliers
      mu_vec    : (n,)   current dual‐slack multipliers
      sigma     : float  centering parameter σ (typical choice 0.1)

    Returns:
      delta_x       : (n,) ndarray  — the primal correction Δx
      delta_lambda  : (m,) ndarray  — the dual (equality) correction Δλ
      delta_mu      : (n,) ndarray  — the dual (slack) correction Δμ
    """

    # Dimensions
    m, n = A.shape[0], A.shape[1]

    # Form X = diag(x)  and  M = diag(mu)
    diagX  = np.diag(x)       # shape (n,n)
    diagMu = np.diag(mu_vec)  # shape (n,n)

    # -----------------------------------------------------------------------
    # Build the Jacobian DF (size (2n + m) × (2n + m)) in block form:
    #
    #     DF = [   0_{n×n}    A^T_{n×m}    I_{n×n}  ]
    #          [   A_{m×n}    0_{m×m}      0_{m×n} ]
    #          [   M_{n×n}    0_{n×m}      X_{n×n}  ]
    #
    jacobian_DF = np.zeros((2*n + m, 2*n + m))

    # Top‐block: [ 0    A^T   I ]
    jacobian_DF[       0 : n,         n : n + m    ] = A.T
    jacobian_DF[       0 : n,       n + m : 2*n + m ] = np.eye(n)

    # Middle‐block: [ A   0   0 ]
    jacobian_DF[   n : n + m,      0 : n       ] = A

    # Bottom‐block: [ M   0   X ]
    jacobian_DF[n + m : 2*n + m,    0 : n       ] = diagMu
    jacobian_DF[n + m : 2*n + m,  n + m : 2*n + m ] = diagX

    # -----------------------------------------------------------------------
    # Compute the three pieces of the residual F(x,λ,μ):
    #   res_dual   = A^T λ + μ − c   ∈ R^n
    #   res_primal = A x − b         ∈ R^m
    #   res_cent   = x ∘ μ           ∈ R^n
    res_dual    = A.T.dot(lambda_vec) + mu_vec - c    # shape (n,)
    res_primal  = A.dot(x) - b                       # shape (m,)
    res_cent    = x * mu_vec                         # shape (n,)

    # τ = (μ^T x) / n
    mu_dot_x = mu_vec.dot(x)
    tau      = mu_dot_x / float(n)   # scalar

    # e = ones(n)
    ones_vector = np.ones(n)

    # Build RHS = −[res_dual; res_primal; res_cent] + [ 0; 0; σ·τ·e ]
    neg_dual   = -res_dual               # length n
    neg_primal = -res_primal             # length m
    cent_shift = -res_cent + sigma * tau * ones_vector  # length n

    RHS = np.concatenate([
        neg_dual,
        neg_primal,
        cent_shift
    ])  # total length = n + m + n = 2n + m

    # -----------------------------------------------------------------------
    # Factor DF once, then solve DF · [Δx; Δλ; Δμ] = RHS
    lu, piv = lu_factor(jacobian_DF)
    solution = lu_solve((lu, piv), RHS)

    # Extract Δx, Δλ, Δμ from the solution vector
    delta_x      = solution[             0 : n       ]  # (n,)
    delta_lambda = solution[         n : n + m       ]  # (m,)
    delta_mu     = solution[ n + m :   2*n + m       ]  # (n,)

    return delta_x, delta_lambda, delta_mu




# Problem 3
def compute_step_lengths(x, mu, dx, dmu):
    """
    Given current x, μ and search directions dx, dμ, returns
    α_max = max α s.t. x + α·dx ≥ 0
    δ_max = max δ s.t. μ + δ·dμ ≥ 0

    All without explicit loops.
    """
    # For dx < 0, candidate α_i = −x_i / dx_i; else +∞
    alpha_cands = np.where(dx < 0, -x / dx, np.inf)
    alpha_max = alpha_cands.min()

    # For dmu < 0, candidate δ_i = −μ_i / dμ_i; else +∞
    delta_cands = np.where(dmu < 0, -mu / dmu, np.inf)
    delta_max = delta_cands.min()

    return alpha_max, delta_max



# Problem 4
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    m, n = A.shape[0], A.shape[1]
    x, lam, mu = starting_point(A, b, c)

    for _ in range(niter):
        # compute search direction
        diagX = np.diag(x)
        diagMu = np.diag(mu)
        DF = np.zeros((2*n + m, 2*n + m))
        DF[0:n,        n:n+m]     = A.T
        DF[0:n,      n+m:2*n+m]   = np.eye(n)
        DF[n:n+m,      0:n]       = A
        DF[n+m:2*n+m,  0:n]       = diagMu
        DF[n+m:2*n+m, n+m:2*n+m]  = diagX

        r_dual   = A.T.dot(lam) + mu - c
        r_pri    = A.dot(x) - b
        r_cent   = x * mu
        tau      = (mu.dot(x)) / float(n)
        ones_vec = np.ones(n)
        RHS = np.concatenate([
            -r_dual,
            -r_pri,
            -r_cent + 0.1 * tau * ones_vec
        ])

        lu, piv = lu_factor(DF)
        soln = lu_solve((lu, piv), RHS)
        dx  = soln[       0 : n      ]
        dlam= soln[     n : n + m    ]
        dmu = soln[n + m : 2*n + m   ]

        # step lengths
        alpha_cand = np.where(dx < 0, -x / dx, np.inf)
        alpha_max  = alpha_cand.min()
        delta_cand = np.where(dmu < 0, -mu / dmu, np.inf)
        delta_max  = delta_cand.min()

        alpha = min(1.0, 0.99 * alpha_max)
        delta = min(1.0, 0.99 * delta_max)

        # update
        x   = x   + alpha * dx
        lam = lam + delta * dlam
        mu  = mu  + delta * dmu

        # duality measure
        nu = x.dot(mu)
        if verbose:
            print("ν =", nu)
        if nu < tol:
            break

    return x, float(c.dot(x))










def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show a scatter‐plot of the data in `filename` together with
    the least‐absolute‐deviations (L1) regression line.

    This function assumes `filename` is a text file with two columns (x, y),
    one point per row, separated by whitespace or commas.
    """

    # (1) Load the data (assume two columns: x_i, y_i)
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    n = len(x)

    # (2) Build the LP to minimize sum |y_i - (m x_i + b)| via variables:
    #       m      : slope
    #       b      : intercept
    #       u_i    : positive part of residual for point i
    #       v_i    : positive part of negative residual for point i
    #
    #    We want:  y_i - (m x_i + b) = u_i - v_i,   with  u_i, v_i >= 0
    #    Equivalently:  m x_i + b + u_i - v_i = y_i
    #
    #    Objective:  minimize ∑ (u_i + v_i).
    #
    #    Decision vector  z  has length 2 + 2n:
    #        z = [ m, b,  u_1, u_2, …, u_n,  v_1, v_2, …, v_n ]^T
    #
    #    c (objective coefficients) = [0, 0, 1, 1, …, 1, 1, 1, …, 1],
    #    where the first two entries (for m, b) are zero, then n ones (for u_i),
    #    then n ones (for v_i).
    #
    #    A_eq and b_eq enforce  n  equalities:  m x_i + b + u_i - v_i = y_i.
    #      - A_eq has shape (n, 2 + 2n).
    #      - Row i of A_eq is:
    #           [ x_i,  1,   0,…,0,  1(at u_i), 0,…,0,  -1(at v_i),  0,…,0 ]
    #        (with the 1 in the u_i column, and -1 in the v_i column).
    #
    #    Bounds:
    #      m and b are free → bounds = (None, None)
    #      u_i, v_i >= 0        → bounds = (0, None)
    #

    # Objective vector
    c_obj = np.hstack([
        np.zeros(2),                 # [0, 0] for (m, b)
        np.ones(n),                  # u_1 … u_n
        np.ones(n)                   # v_1 … v_n
    ])  # length = 2 + 2n

    # Equality constraints: A_eq · z = b_eq
    # Build A_eq row by row
    A_eq = np.zeros((n, 2 + 2*n))
    b_eq = y.copy()  # each entry is y_i

    for i in range(n):
        # m coefficient
        A_eq[i, 0] = x[i]
        # b coefficient
        A_eq[i, 1] = 1.0
        # u_i coefficient  (+1)
        A_eq[i, 2 + i] = 1.0
        # v_i coefficient  (−1)
        A_eq[i, 2 + n + i] = -1.0

    # Bounds for each variable in z
    bounds = []
    # m (unbounded)
    bounds.append((None, None))
    # b (unbounded)
    bounds.append((None, None))
    # u_i >= 0
    for _ in range(n):
        bounds.append((0, None))
    # v_i >= 0
    for _ in range(n):
        bounds.append((0, None))

    # Solve the LP via linprog
    res = linprog(
        c=c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
        options={'disp': False}
    )

    if not res.success:
        raise RuntimeError(f"Linear‐program solver failed: {res.message}")

    # Extract the optimal slope m* and intercept b*
    m_star = res.x[0]
    b_star = res.x[1]

    # (3) Produce the scatter plot and draw the L1 regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points', zorder=3)

    # For plotting the line, choose two x‐values spanning the data range
    x_min, x_max = x.min(), x.max()
    x_line = np.array([x_min, x_max])
    y_line = m_star * x_line + b_star

    plt.plot(x_line, y_line, color='red', linewidth=2, label='L1 regression')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least‐Absolute‐Deviations Regression')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    
    
    
    if __name__ == "__main__":
    # For reproducibility
    np.random.seed(0)

    # (1) Generate a random LP with j constraints and k “true” variables.
    #     The remaining j variables are slack variables, so A has shape (j, j+k).
    j, k = 5, 3
    A, b, c, x_true = randomLP(j, k)

    # (2) Call your interior‐point solver.
    #     Increase niter/tighten tol as needed for convergence.
    x_opt, val = interiorPoint(A, b, c, niter=50, tol=1e-8, verbose=True)

    # (3) Print results
    print("\n=== Interior‐Point Result ===")
    print("Computed x*:", x_opt)
    print("Objective value c^T x*:", val)

    # (4) Quick feasibility check: ||A x* − b||
    primal_residual = np.linalg.norm(A.dot(x_opt) - b)
    print("Primal feasibility (‖Ax* − b‖):", primal_residual)

    # (5) Compare c^T x* to c^T x_true (if you want to see how close you got
    #     to the hidden “true” solution x_true that randomLP returned).
    true_obj = c.dot(x_true)
    print("c^T x_true (ground‐truth):", true_obj)
    print("Difference |c^T x* − c^T x_true|:", abs(val - true_obj))
    
    
    
    np.random.seed(0)
    j, k = 5, 3
    A, b, c, x_true = randomLP(j, k)

    x_opt, val = interiorPoint(A, b, c, niter=50, tol=1e-8, verbose=False)

    print("x* =", x_opt)
    print("c^T x* =", val)
    print("‖A x* − b‖ =", np.linalg.norm(A.dot(x_opt) - b))
    print("c^T x_true =", c.dot(x_true))
    print("|c^T x* − c^T x_true| =", abs(val - c.dot(x_true)))
    
    
    
    
    
    x   = np.array([0.5, 2.0, 0.1, 1.5])
    mu  = np.array([1.0, 0.2, 3.0, 0.5])
    dx  = np.array([-0.4, 0.1, -0.05, 0.2])
    dmu = np.array([ 0.3, -0.1, -0.5, 0.4])

    alpha_max, delta_max = compute_step_lengths(x, mu, dx, dmu)

    print("alpha_max =", alpha_max)
    print("delta_max =", delta_max)
    
    
    
    
    
    
    np.random.seed(0)
    j, k = 7, 5
    A, b, c, x_true = randomLP(j, k)

    x_opt, val = interiorPoint(A, b, c)

    print("x* =", x_opt)
    print("c^T x* =", val)
    print("‖A x* − b‖ =", np.linalg.norm(A.dot(x_opt) - b))
    print("c^T x_true =", c.dot(x_true))
    print("|c^T x* − c^T x_true| =", abs(val - c.dot(x_true)))
    
    
    
    
    
    
    leastAbsoluteDeviations('simdata.txt')
    
    
    
    

