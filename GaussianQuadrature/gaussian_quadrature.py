# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Emmanuel Oguadimma 
MTH 520
30/05/2025
"""

import numpy as np
from scipy.stats import norm
from scipy import linalg as la
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.integrate import quad

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial. Also store the inverse
        weight function w(x)⁻¹ = 1/w(x).

        Parameters:
            n (int): Number of quadrature points.
            polytype (str): Either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # Validate inputs
        if polytype not in ("legendre", "chebyshev"):
            raise ValueError("polytype must be 'legendre' or 'chebyshev'")

        self.n = int(n)
        self.polytype = polytype

        # Define the weight function w(x) for the chosen family:
        if polytype == "legendre":
            # Legendre weight is w(x) = 1 on [-1,1]
            self.w = lambda x: np.ones_like(x, dtype=float)
        else:
            # Chebyshev weight is w(x) = 1/√(1 − x²) on (-1,1)
            self.w = lambda x: 1.0 / np.sqrt(1.0 - x**2)

        # Store the reciprocal weight w(x)^(-1)
        self.w_inv = lambda x: 1.0 / self.w(x)

        # placeholders for the quadrature nodes and weights (Problem 3)
        self.points = None
        self.weights = None
        
        
      

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # Build recurrence coefficients for the chosen family
        # α_k = 0 for both Legendre and Chebyshev
        alpha = np.zeros(n, dtype=float)

        # β_k for k = 1,…,n-1
        if self.polytype == "legendre":
            # β_k = k^2 / (4 k^2 − 1)
            k = np.arange(1, n)
            beta = (k**2) / (4.0 * k**2 - 1.0)
            m0 = 2.0                           # ∫_{-1}^1 w(x) dx for Legendre
        else:  # chebyshev (first kind)
            # β_k = 1/2  for all k ≥ 1
            beta = 0.5 * np.ones(n-1, dtype=float)
            m0 = np.pi                         # ∫_{-1}^1 dx/√(1−x²) = π

        # Assemble the symmetric Jacobi matrix J
        J = np.diag(alpha)
        off = np.sqrt(beta)
        J += np.diag(off, 1) + np.diag(off, -1)

        # Compute eigenvalues (points) and eigenvectors
        points, V = eigh(J)

        # Weights: w_i = m0 * [v₀_i]², where v₀_i is the first component of the i-th eigenvector
        weights = m0 * (V[0, :]**2)

        # Save on the instance
        self.points  = points
        self.weights = weights

        return points, weights
        
        
        
        

    # Problem 3
     def basic(self, f):
        """Approximate the integral of f over [-1, 1] using the stored
        Gaussian quadrature nodes and weights.

        The integral
            ∫_{-1}^1 f(x) dx
        is approximated by
            ∑_{i=1}^n w_i · g(x_i),
        where g(x) = f(x)/w(x), and w_i are the quadrature weights.

        Parameters:
            f (callable or array‐like): A function f(x) defined on [-1,1].

        Returns:
            float: The quadrature approximation to ∫_{-1}^1 f(x) dx.
        """
        # Ensure we have nodes and weights
        if self.points is None or self.weights is None:
            self.points, self.weights = self.points_weights(self.n)

        x = self.points            # nodes x_i
        w = self.weights          # quadrature weights w_i

        # Evaluate f at the nodes (works if f is a vectorized function)
        fx = np.array([f(xi) for xi in x])

        # Compute g(x) = f(x) / w(x) via the stored w_inv
        gx = fx * self.w_inv(x)

        # Weighted sum
        return float(np.dot(w, gx))
        
        
        

    # Problem 4
        def integrate(self, f, a, b):
        """Approximate the integral of f over [a, b] using Gaussian quadrature.

        Transforms the integral to [-1,1] via the change of variables
            x = (b−a)/2 · t + (b+a)/2,
        so that
            ∫_a^b f(x) dx = (b−a)/2 ∫_{−1}^1 f((b−a)/2·t + (b+a)/2) dt.
        We then apply our basic() method to the integrand h(t) = f((b−a)/2·t + (b+a)/2).

        Parameters:
            f (callable): Function f(x) to integrate.
            a (float): Lower limit.
            b (float): Upper limit.

        Returns:
            float: Approximate value of ∫_a^b f(x) dx.
        """
        # scaling and shift constants
        scale = (b - a) / 2.0
        shift = (b + a) / 2.0

        # define h(t) = f(scale*t + shift)
        def h(t):
            return f(scale * t + shift)

        # approximate ∫_{-1}^1 h(t) dt via basic()
        integral_on_minus1_to_1 = self.basic(h)

        # scale back to [a,b]
        return scale * integral_on_minus1_to_1

        
        
        

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two‐dimensional function f on
        the rectangle [a1,b1]×[a2,b2] via nested Gaussian quadrature.

        Uses the identity
            ∬ f(x,y) dx dy
          = ∫_{y=a2}^{b2} [ ∫_{x=a1}^{b1} f(x,y) dx ] dy
          = ∫_{a2}^{b2} h(y) dy,
        where h(y) = ∫_{a1}^{b1} f(x,y) dx.

        Parameters:
            f  (callable):    a function of two variables, f(x,y).
            a1 (float), b1:   x–integration limits.
            a2 (float), b2:   y–integration limits.

        Returns:
            float: the approximate value of ∬_{[a1,b1]×[a2,b2]} f(x,y) dx dy.
        """
        # ensure we’ve computed nodes & weights
        if self.points is None or self.weights is None:
            self.points_weights(self.n)

        # inner integral: for a fixed y, integrate in x
        def h(y):
            return self.integrate(lambda x: f(x, y), a1, b1)

        # outer integral: integrate h(y) in y
        return self.integrate(h, a2, b2)
    

# Problem 5
def prob5():
    """Experiment with Gaussian quadrature errors integrating the standard normal
    density f(x) = (1/√(2π)) e^{–x²/2} from –3 to 2, for n = 5,10,…,50."""
    # 1) Exact value via the normal CDF
    F_exact = norm.cdf(2) - norm.cdf(-3)

    # 2) Reference error of scipy.integrate.quad
    f = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    F_quad, _ = quad(f, -3, 2)
    err_quad = abs(F_quad - F_exact)

    # 3) Prepare arrays of n values and storage for errors
    ns = np.arange(5, 51, 5)
    err_leg = []
    err_cheb = []

    for n in ns:
        # --- Legendre rule ---
        G_leg = GaussianQuadrature(n, polytype="legendre")
        G_leg.points_weights(n)
        F_leg = G_leg.integrate(f, -3, 2)
        err_leg.append(abs(F_leg - F_exact))

        # --- Chebyshev rule ---
        G_cheb = GaussianQuadrature(n, polytype="chebyshev")
        G_cheb.points_weights(n)
        F_cheb = G_cheb.integrate(f, -3, 2)
        err_cheb.append(abs(F_cheb - F_exact))

    # 4) Plot errors vs n
    plt.figure(figsize=(8,5))
    plt.semilogy(ns, err_leg, 'o-', label='Legendre')
    plt.semilogy(ns, err_cheb, 's-', label='Chebyshev')
    plt.hlines(err_quad, ns[0], ns[-1], colors='gray', linestyles='--',
               label=f'quad error ≈ {err_quad:.2e}')

    plt.xlabel('Number of points $n$')
    plt.ylabel('Absolute error')
    plt.title('Gaussian Quadrature Error for ∫_{-3}^2 φ(x)dx')
    plt.legend()
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.show()

    
    
    
    if __name__ == "__main__":
    # Instantiate a 5‐point Legendre rule
    G = GaussianQuadrature(5, polytype="legendre")
    # Check the weight and its reciprocal at x=0
    print("w(0) =", G.w(0), ", w_inv(0) =", G.w_inv(0))

    # Instantiate a 4‐point Chebyshev rule
    C = GaussianQuadrature(4, polytype="chebyshev")
    print("w(0.5) =", C.w(0.5), ", w_inv(0.5) =", C.w_inv(0.5))
    
    G5 = GaussianQuadrature(5, polytype="legendre")
x, w = G5.points_weights(5)
print("Nodes:", x)
print("Weights:", w)

f = lambda x: 1/np.sqrt(1 - x**2)

# Use Chebyshev rule for this weight function
G = GaussianQuadrature(50, polytype="chebyshev")
# compute nodes & weights
G.points_weights(50)
approx = G.basic(f)

# Compare to scipy.integrate.quad
true, _ = quad(f, -1, 1)
print("Gaussian quadrature approx:", approx)
print("Reference (quad):", true)

G = GaussianQuadrature(5)
    G.points_weights(5)
    result = G.integrate(np.sin, 0, np.pi)
    print(result)
    
    f = lambda x, y: np.sin(x) + np.cos(y)

    # 2) Instantiate a quadrature rule (e.g. 20‐point Legendre)
    G = GaussianQuadrature(20, polytype="legendre")
    G.points_weights(20)

    # 3) Approximate the double integral over [-10,10]×[-1,1]
    approx = G.integrate2d(f, -10, 10, -1, 1)

    # 4) Compare to scipy.integrate.nquad
    exact, _ = nquad(f, [[-10,10], [-1,1]])

    print(f"GaussianQuadrature approx: {approx:.8f}")
    print(f"scipy.integrate.nquad exact: {exact:.8f}")
    print(f"Error: {abs(approx - exact):.2e}")
    
    
    prob5()

