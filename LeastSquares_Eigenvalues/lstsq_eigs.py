# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Emmanuel Oguadimma 
MTH 520
09/06/2025
"""

import numpy as np
from cmath import sqrt
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solution to Ax = b using the QR decomposition.

    Parameters:
        A ((m, n) ndarray): A matrix of full column rank n ≤ m.
        b ((m,) ndarray):   A vector of length m.

    Returns:
        x ((n,) ndarray):   The minimizer of ||Ax − b||₂, i.e. the solution to R x = Qᵀb.
    """
    # Compute the thin (reduced) QR decomposition of A: A = Q @ R
    # Q is m×n with orthonormal columns, R is n×n upper triangular.
    Q, R = np.linalg.qr(A, mode='reduced')
    
    # Compute Qᵀ b (an n-vector)
    y = Q.T @ b
    
    # Solve R x = y for x (since R is upper triangular and invertible)
    x = np.linalg.solve(R, y)
    
    return x


# Problem 2
def line_fit():
    """Find the least squares line that relates year to the housing price index
    for the data in housing.npy, and plot both the data points and the fit."""
    # Load the data: each row is [year, index]
    data = np.load("housing.npy")
    years = data[:, 0]
    hpi   = data[:, 1]

    # Build the design matrix for y = m*x + c
    A = np.vstack([years, np.ones_like(years)]).T

    # Solve for [m, c] in the least-squares sense
    (m, c), *_ = np.linalg.lstsq(A, hpi, rcond=None)

    # Plot the raw data
    plt.figure()
    plt.scatter(years, hpi, label="Data", color="tab:orange", marker="x")

    # Plot the best‐fit line
    x_line = np.array([years.min(), years.max()])
    y_line = m * x_line + c
    plt.plot(x_line, y_line, label=f"Fit: y = {m:.2f}x + {c:.2f}", color="tab:blue")

    plt.xlabel("Year")
    plt.ylabel("Housing Price Index")
    plt.title("Least Squares Fit of Housing Price Index vs. Year")
    plt.legend()
    plt.grid(True)
    plt.show()



# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy.
    Plot both the data points and the least squares polynomials in individual subplots.
    """
    # Load data: each row is [year, housing_price_index]
    data = np.load("housing.npy")
    years = data[:, 0]
    hpi   = data[:, 1]

    # Degrees to fit
    degrees = [3, 6, 9, 12]

    # Prepare a 2×2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, deg in zip(axes.ravel(), degrees):
        # Compute least-squares polynomial coefficients
        coeffs = np.polyfit(years, hpi, deg)
        poly   = np.poly1d(coeffs)

        # Generate a dense set of x values for a smooth curve
        x_line = np.linspace(years.min(), years.max(), 300)
        y_line = poly(x_line)

        # Plot data and fit
        ax.scatter(years, hpi, label="Data", color="tab:orange", marker="x")
        ax.plot(x_line, y_line, label=f"Degree {deg}", color="tab:blue")

        ax.set_title(f"Degree {deg} Polynomial Fit")
        ax.set_xlabel("Year")
        ax.set_ylabel("Housing Price Index")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    # Load the (x, y) data
    data = np.load("ellipse.npy")
    x, y = data[:, 0], data[:, 1]

    # Build the design matrix for ax² + bx + cxy + dy + ey² = 1
    D = np.vstack([x**2, x, x*y, y, y**2]).T
    ones = np.ones_like(x)

    # Solve the least-squares problem D @ [a,b,c,d,e] = 1
    (a, b, c, d, e), *_ = np.linalg.lstsq(D, ones, rcond=None)

    # Plot the raw data
    plt.figure()
    plt.scatter(x, y, color="tab:orange", marker="x", label="Data")

    # Overlay the fitted ellipse
    plot_ellipse(a, b, c, d, e)
    plt.title("Least-Squares Ellipse Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect("equal", "datalim")
    plt.show()




# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    # Start with a random vector of unit length
    n = A.shape[0]
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    lambda_old = 0.0

    for _ in range(N):
        # Multiply and re-normalize
        w = A @ v
        v = w / np.linalg.norm(w)
        # Rayleigh quotient for eigenvalue estimate
        lambda_ = v @ (A @ v)
        # Check convergence
        if abs(lambda_ - lambda_old) < tol:
            break
        lambda_old = lambda_

    return lambda_, v



# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the (unshifted) QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of QR iterations to run.
        tol (float): Threshold for detecting when a subdiagonal entry is effectively zero,
                     indicating a 1×1 (real) block vs. a 2×2 (complex-conjugate) block.

    Returns:
        ((n,) ndarray): The computed eigenvalues of A.
    """
    # Work on a copy in floating-point
    S = A.astype(float).copy()
    n = S.shape[0]

    # Perform N iterations of S ← R @ Q from the QR decomposition
    for _ in range(N):
        Q, R = np.linalg.qr(S)
        S = R @ Q

    # After convergence, S is (approximately) quasi-triangular:
    # real eigenvalues on the diagonal, and any complex pairs in 2×2 blocks.
    eigs = []
    i = 0
    while i < n:
        # If we're at the last row, only a 1×1 block remains
        if i == n - 1:
            eigs.append(S[i, i])
            i += 1
        else:
            # Check the subdiagonal entry
            if abs(S[i+1, i]) < tol:
                # 1×1 real block
                eigs.append(S[i, i])
                i += 1
            else:
                # 2×2 block for a complex-conjugate pair
                block = S[i:i+2, i:i+2]
                # Compute its eigenvalues
                evals = np.linalg.eigvals(block)
                eigs.extend(evals)
                i += 2

    return np.array(eigs)
    

    
    
    if __name__ == "__main__":
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)
    b = np.array([7, 8, 9], dtype=float)

    x_ls = least_squares(A, b)
    print("x =", x_ls)
    # Check residual norm
    print("||Ax - b|| =", np.linalg.norm(A @ x_ls - b))
    
    line_fit()
    polynomial_fit()
    ellipse_fit()


    A = np.array([[2, 1],
                  [1, 3]], dtype=float)
    dominant_eigval, eigvec = power_method(A)
    print("Dominant eigenvalue:", dominant_eigval)
    print("Corresponding eigenvector:", eigvec)
    
    eigs = qr_algorithm(A)
    print("Computed eigenvalues:", eigs)
