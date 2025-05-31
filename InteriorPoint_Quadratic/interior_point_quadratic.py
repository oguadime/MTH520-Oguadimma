# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
Emmanuel Oguadimma
MTH 520
30/01/2025
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
from scipy.sparse import spdiags


import numpy as np
from scipy import linalg as la

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
        min (1/2) x^T G x + c^T x
        s.t. A x >= b
    using the provided “guess” = (x0, y0, l0).

    Returns a tuple (x0, y0, l0) where y0 > 0 and l0 > 0.
    """
    x0, y0, l0 = guess
    m, n = A.shape

    # Build the linear system N · [dx; dy; dl] = rhs to enforce feasibility
    N = np.zeros((n + m + m, n + m + m))
    # Top‐left block: G  (n×n)
    N[:n, :n] = G
    # Top‐right (blocks): −A^T in columns (n+m : n+m+m)
    N[:n, n + m:] = -A.T
    # Middle block: A (rows n : n+m, columns 0 : n)
    N[n:n + m, :n] = A
    # Middle‐middle: −I_m (rows n : n+m, columns n : n+m)
    N[n:n + m, n:n + m] = -np.eye(m)
    # Bottom blocks:   diag(l0) at (rows n+m : n+m+m, cols n : n+m)
    N[n + m : n + m + m, n : n + m]     = np.diag(l0)
    #                 diag(y0) at (rows n+m : n+m+m, cols n+m : n+m+m)
    N[n + m : n + m + m, n + m : n + m + m] = np.diag(y0)

    # Build rhs = [−(G x0 − A^T l0 + c);
    #              −(A x0 − y0 − b);
    #              −(y0 ∘ l0)]
    rhs = np.empty(n + m + m)
    rhs[:n]       = -(G.dot(x0) - A.T.dot(l0) + c)
    rhs[n:n + m]  = -(A.dot(x0) - y0 - b)
    rhs[n + m:]   = -(y0 * l0)

    # Solve for corrections (dx, dy, dl)
    sol = la.solve(N, rhs)
    dy = sol[n:n + m]
    dl = sol[n + m:]

    # Shift y0, l0 so they stay positive and not too small
    y0 = np.maximum(1.0, np.abs(y0 + dy))
    l0 = np.maximum(1.0, np.abs(l0 + dl))

    return x0, y0, l0


def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """
    Solve the QP
        minimize   (1/2) x^T Q x  +  c^T x
        subject to A x >= b
    via a primal–dual interior‐point method.

    Parameters:
      Q       : (n×n) positive semidefinite matrix
      c       : (n,)   objective vector
      A       : (m×n)  inequality‐constraint matrix
      b       : (m,)   RHS vector
      guess   : (x0, y0, l0) initial guesses, arrays of lengths (n, m, m).
                y0 and l0 must be strictly positive.
      niter   : maximum number of Newton iterations
      tol     : tolerance on the complementarity measure τ
      verbose : if True, print ‖F‖ and τ each iteration

    Returns:
      x_opt : (n,) the optimal primal solution
      val   : float the final objective value ½ x_opt^T Q x_opt + c^T x_opt
    """
    # Unpack dimensions and initial guesses
    x, y, l = startingPoint(Q, c, A, b, guess)
    m, n = A.shape[0], A.shape[1]

    for k in range(niter):
        # -----------------------------------------------
        # (1) Compute residual blocks F(x,y,l):
        #     res_stationarity = Q x + c − A^T l       ∈ R^n
        #     res_primal       = A x − y − b           ∈ R^m
        #     res_complement   = y ∘ l                 ∈ R^m
        res_stationarity = Q.dot(x) + c - A.T.dot(l)   # shape (n,)
        res_primal       = A.dot(x) - y - b            # shape (m,)
        res_complement   = y * l                       # shape (m,)

        # Stack into single (n + m + m,) vector
        F = np.concatenate([res_stationarity, res_primal, res_complement])

        # -----------------------------------------------
        # (2) Build Jacobian DF(x,y,l) of size (n+m+m)×(n+m+m):
        #      DF = [  Q            0        −A^T    ]
        #           [  A          −I_m        0     ]
        #           [  0      diag(l)    diag(y)   ]
        DF = np.zeros((n + m + m, n + m + m))

        # Top‐row blocks:
        DF[          0 : n,          0 : n       ] = Q                # ∂res_stationarity/∂x
        DF[          0 : n,    n + m : n + m + m   ] = -A.T             # ∂res_stationarity/∂l

        # Middle‐row blocks:
        DF[    n : n + m,         0 : n            ] = A                # ∂res_primal/∂x
        DF[    n : n + m,       n : n + m          ] = -np.eye(m)       # ∂res_primal/∂y

        # Bottom‐row blocks:
        DF[n + m : n + m + m,    n : n + m          ] = np.diag(l)       # ∂res_complement/∂y
        DF[n + m : n + m + m, n + m : n + m + m      ] = np.diag(y)       # ∂res_complement/∂l

        # -----------------------------------------------
        # (3) Compute complementarity measure τ = (y^T l)/m, and σ = 0.1
        tau = (y.dot(l)) / float(m)
        sigma = 0.1

        # -----------------------------------------------
        # (4) Form RHS = −F + [0; 0; σ · τ · e], where e = ones(m)
        RHS = -F.copy()
        RHS[n + m : n + m + m] += sigma * tau * np.ones(m)

        # -----------------------------------------------
        # (5) Solve DF · [Δx; Δy; Δl] = RHS
        delta = la.solve(DF, RHS)
        delta_x      = delta[                0 : n       ]   # shape (n,)
        delta_y      = delta[              n : n + m      ]   # shape (m,)
        delta_l      = delta[    n + m : n + m + m          ]   # shape (m,)

        # -----------------------------------------------
        # (6) Compute step length α so that y + α·Δy ≥ 0 and l + α·Δl ≥ 0.
        #     α_max_y = min_i { −y_i / Δy_i  |  Δy_i < 0 }, otherwise +inf
        #     α_max_l = min_i { −l_i / Δl_i  |  Δl_i < 0 }, otherwise +inf
        #     α = min(1, 0.99·α_max_y, 0.99·α_max_l)
        alpha_candidates_y = np.where(delta_y < 0, -y / delta_y, np.inf)
        alpha_candidates_l = np.where(delta_l < 0, -l / delta_l, np.inf)
        alpha_max_y = alpha_candidates_y.min()
        alpha_max_l = alpha_candidates_l.min()
        alpha = min(1.0, 0.99 * alpha_max_y, 0.99 * alpha_max_l)

        # -----------------------------------------------
        # (7) Update variables:
        #     x   ← x   + α · Δx
        #     y   ← y   + α · Δy
        #     l   ← l   + α · Δl
        x = x + alpha * delta_x
        y = y + alpha * delta_y
        l = l + alpha * delta_l

        if verbose:
            normF = np.linalg.norm(F)
            print(f"iter {k:2d}  ‖F‖ = {normF:.3e}  τ = {tau:.3e}  α = {alpha:.3e}")

        # (8) Check convergence: stop if τ < tol
        if tau < tol:
            break

    # After convergence (or hitting max iterations), return x_opt and objective
    x_opt = x
    val = 0.5 * x_opt.dot(Q.dot(x_opt)) + c.dot(x_opt)
    return x_opt, float(val)






def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n×n grid."""
    data = -1 * np.ones((5, n**2))
    data[2, :] = 4
    # Zero out wrap‐around on row boundaries
    data[1, n-1::n] = 0
    data[3, ::n]   = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

def circus(n=15):
    """
    Solve the “circus tent” problem on an n×n grid:
      • Find the surface z (an array of length n²) that minimizes ½ z^T H z − 1^T z
        subject to z ≥ f_stakes and z(boundary) = 0.
      • Here H is the discrete Laplacian matrix (Dirichlet energy), and
        f_stakes is a vector of “stake” heights at selected interior nodes.
    After solving, display z as a 3D surface.

    (If you have your own stake locations & heights, replace the code in
    “define f_full” with your data.)
    """

    # (1) Build the full Laplacian H of size (n² × n²)
    H_full = laplacian(n)

    # (2) Identify boundary vs. interior indices
    #     We index grid nodes in row‐major order: index = i*n + j (0 ≤ i,j < n)
    all_indices = np.arange(n**2)
    boundary_mask = np.zeros(n**2, dtype=bool)

    # Mark top and bottom rows
    boundary_mask[0 : n] = True
    boundary_mask[n*(n-1) : n*n] = True
    # Mark left and right columns
    boundary_mask[0 : n**2 : n] = True
    boundary_mask[n-1 : n**2 : n] = True

    interior_mask = ~boundary_mask
    interior_idx = all_indices[interior_mask]   # indices of interior nodes
    m_int = interior_idx.size                   # number of interior variables

    # (3) Build H_int = H_full[interior, interior]
    H_int = H_full[np.ix_(interior_idx, interior_idx)]

    # (4) Define the “stake” heights f_full for all n² nodes, then extract f_int
    #
    # For illustration, we place a few stakes at chosen interior grid points.
    # You can replace this block with your own stakes data if available.
    f_full = np.zeros(n**2)

    # Example: place three stakes inside the grid at given (i,j) with heights:
    #    stake 1 at (i= n//3, j= n//3), height = 2.0
    #    stake 2 at (i= n//2, j= n//2), height = 3.0
    #    stake 3 at (i= 2n//3, j= n//4), height = 1.5
    stakes = [
        (n//3,     n//3,     2.0),
        (n//2,     n//2,     3.0),
        (2*n//3,   n//4,     1.5)
    ]
    for (i, j, h) in stakes:
        idx = i * n + j
        f_full[idx] = h

    # Extract only the interior stake‐height vector f_int
    f_int = f_full[interior_mask]   # length = m_int

    # (5) Set up QP:  minimize (1/2) z_int^T H_int z_int  −  1^T z_int
    #     subject to z_int ≥ f_int.
    #
    # In cvxopt notation:  minimize (1/2) x^T P x + q^T x,  s.t. G x ≤ h.
    #
    # Here P = H_int (must be positive semidef),  q = −1 (vector of length m_int).
    # We want z ≥ f_int  ↔  −z ≤ −f_int.  So G = −I,  h = −f_int.
    #
    P = matrix(H_int)                              # (m_int × m_int)
    q = matrix(-np.ones((m_int, 1)))               # length m_int
    G = matrix(-np.eye(m_int))                     # enforce z_int ≥ f_int
    h = matrix(-f_int.reshape((m_int, 1)))

    # (6) Solve the QP using cvxopt.solvers.qp
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    z_int = np.array(sol['x']).flatten()           # the interior solution

    # (7) Reconstruct the full grid solution z_full (length n²)
    z_full = np.zeros(n**2)
    # Boundary nodes remain 0 (tent is pegged to 0 at boundary)
    z_full[interior_mask] = z_int

    # (8) Reshape z_full into an (n × n) array for plotting
    Z = z_full.reshape((n, n))

    # (9) Display as a 3D surface
    x_coords = np.linspace(0, 1, n)
    y_coords = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f"Circus Tent Solution on {n}×{n} Grid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (tent height)")
    plt.tight_layout()
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """
    Markowitz Portfolio Optimization.

    Parameters:
        filename (str): Name of the text file containing historical returns.
                        First column is years (ignored), next 8 columns are asset returns.

    Returns:
        x_short   : (8,) ndarray, optimal weights allowing short selling.
        x_noshort: (8,) ndarray, optimal weights disallowing short selling.
    """

    # (1) Load data: skip the first column of years, keep only the 8 asset‐return columns.
    data = np.loadtxt(filename)
    returns = data[:, 1:]               # shape = (T, 8)

    # (2) Compute expected returns μ (sample mean) and covariance Q (sample covariance).
    mu     = np.mean(returns, axis=0)   # shape = (8,)
    Q_np   = np.cov(returns, rowvar=False)  # shape = (8, 8)

    # (3) Set target return R. Enforce μ^T x = R and ∑ x_i = 1.
    R = 1.13

    # (4) Convert Q and other vectors/matrices to CVXOPT format.
    P = matrix(Q_np)                          # (8×8) matrix
    q = matrix(np.zeros((8, 1)))              # zero linear term (we just minimize variance)

    # Equality constraints: A_eq x = b_eq, where
    #   A_eq = [ μ^T ;  1^T ]  (2×8),
    #   b_eq = [ R ;      1 ]  (2×1).
    A_eq = matrix(np.vstack([mu, np.ones(8)]))
    b_eq = matrix(np.array([R, 1.0]).reshape((2, 1)))

    # --------------------------------------------------------
    # (5) Solve with short selling allowed: x ∈ R^8 (no sign constraints).
    #    In CVXOPT we can pass empty G, h for no inequalities.
    G_free = matrix(np.zeros((0, 8)))
    h_free = matrix(np.zeros((0, 1)))

    sol_free = solvers.qp(P, q, G_free, h_free, A_eq, b_eq)
    x_short_opt = np.array(sol_free['x']).flatten()  # shape = (8,)

    # --------------------------------------------------------
    # (6) Solve with no short selling: enforce x ≥ 0  ↔  −x ≤ 0.
    G_noshort = matrix(-np.eye(8))               # (8×8) so that G_noshort · x ≤ h_noshort enforces x ≥ 0
    h_noshort = matrix(np.zeros((8, 1)))

    sol_noshort = solvers.qp(P, q, G_noshort, h_noshort, A_eq, b_eq)
    x_noshort_opt = np.array(sol_noshort['x']).flatten()

    return x_short_opt, x_noshort_opt

    
    
    
    
    
    if __name__ == "__main__":
    import numpy as np

    # Example QP:  min (1/2)x^T x  subject to  x >= 1
    n = 3
    Q = np.eye(n)
    c = np.zeros(n)
    A = np.eye(n)                  # Ax >= b  ↔  x_i >= 1
    b = np.ones(n)

    # Initial guess: x0 > 1 so slack = x0 - 1 > 0, and mu0 > 0
    x0     = 2.0 * np.ones(n)
    slack0 = x0 - b                 # = ones(n)
    mu0    = np.ones(n)

    delta_x, delta_slack, delta_mu = qInteriorPoint(
        Q, c, A, b, (x0, slack0, mu0), verbose=True
    )

    print("Δx      =", delta_x)
    print("Δslack  =", delta_slack)
    print("Δmu     =", delta_mu)

    
    
    
    Q = np.array([[1.0, -1.0],
                  [-1.0,  2.0]])
    c = np.array([-2.0, -6.0])
    A = np.array([
        [-1.0, -1.0],
        [ 1.0, -2.0],
        [-2.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0]
    ])
    b = np.array([-2.0, -2.0, -3.0, 0.0, 0.0])

    # Initial guess: x0 = [0.5, 0.5], y0 = A x0 − b, l0 = ones(m)
    x0 = np.array([0.5, 0.5])
    y0 = A.dot(x0) - b               # slack = Ax0 − b, must be > 0
    l0 = np.ones(A.shape[0])         # mu = ones

    x_opt, obj_val = qInteriorPoint(Q, c, A, b, (x0, y0, l0), verbose=True)

    print("\nOptimal x =", x_opt)
    print("Optimal objective value =", obj_val)
    # Expected minimizer is [2/3, 4/3]. 
    # The printed result should be very close to [0.666..., 1.333...].\\
    
    
    
    circus(n=15)
    
    
     x_short, x_noshort = portfolio("portfolio.txt")
    print("Optimal with short selling:   ", x_short)
    print("Optimal without short selling:", x_noshort)