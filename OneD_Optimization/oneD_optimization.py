# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Emmanuel Oguadimma 
MTH 520
30/05/2025
"""

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from scipy.optimize import newton as sp_newton

# Problem 1
def newton(f, x0, Df, tol=1e-5, maxiter=15):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (callable): R → R (or R^n → R^n).
        x0 (float or ndarray): Initial guess.
        Df (callable): Derivative of f (Jacobian for n>1).
        tol (float): Tolerance for stopping criterion.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (float or ndarray): Final approximation to a root of f.
        converged (bool): True if |x_k − x_{k−1}| < tol within maxiter.
        k (int): Number of iterations performed.
    """
    x = x0
    for k in range(1, maxiter + 1):
        fx = f(x)
        Dfx = Df(x)
        # Avoid division by zero / singular Jacobian
        if Dfx == 0 or (hasattr(Dfx, 'ndim') and np.linalg.cond(Dfx) > 1/np.finfo(float).eps):
            return x, False, k-1

        # Newton step
        x_new = x - fx / Dfx

        # Check convergence
        if np.all(np.abs(x_new - x) < tol):
            return x_new, True, k

        x = x_new

    # If we exit the loop, we did not converge within maxiter
    return x, False, maxiter




# Problem 2
def plot_basins(f, Df, zeros, domain, res=400, iters=20):
    """Plot the basins of attraction of f on the complex plane via Newton’s method.

    Parameters:
        f (callable):     function C→C.
        Df (callable):    its derivative C→C.
        zeros (ndarray):  1-D array of known roots of f.
        domain (list):    [r_min, r_max, i_min, i_max] bounds in the complex plane.
        res (int):        resolution per axis (res×res grid).
        iters (int):      number of Newton iterations to apply.
    """
    r_min, r_max, i_min, i_max = domain

    # 1) build the initial complex grid X0 of shape (res, res)
    re = np.linspace(r_min, r_max, res)
    im = np.linspace(i_min, i_max, res)
    Re, Im = np.meshgrid(re, im)
    X = Re + 1j * Im  # this is X₀

    # 2) apply Newton’s method iters times (vectorized)
    for _ in range(iters):
        X = X - f(X) / Df(X)

    # 3) assign each converged point to the nearest known zero
    #    compute distances to each root: shape (len(zeros), res, res)
    diff = X[np.newaxis, :, :] - zeros[:, np.newaxis, np.newaxis]
    idx = np.abs(diff).argmin(axis=0)  # shape (res, res), values in [0, R-1]

    # 4) plot with pcolormesh
    plt.figure(figsize=(6,6))
    # pcolormesh wants x coords of shape (res,) or (res, res+1); here using centers:
    pcm = plt.pcolormesh(Re, Im, idx, cmap='brg', shading='auto')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'Basins of attraction ({iters} Newton steps)')
    plt.colorbar(pcm, ticks=range(len(zeros)), label='root index')
    plt.show()
    
    
    


# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=100):
    """
    Use the secant method on df to find a stationary point (minimizer) of f.

    Parameters:
        df (callable): The derivative f′(x).
        x0 (float):    First initial guess.
        x1 (float):    Second initial guess.
        tol (float):   Stopping tolerance on |x_{k} – x_{k−1}|.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (float):     Approximate stationary point.
        converged (bool): True if |x_k − x_{k−1}| < tol before maxiter.
        k (int):       Number of iterations performed.
    """
    # Evaluate derivative at the two starting points
    df0 = df(x0)
    df1 = df(x1)

    for k in range(1, maxiter + 1):
        # Denominator of the secant update
        denom = df1 - df0
        if denom == 0:
            # derivative difference zero ⇒ cannot proceed
            return x1, False, k-1

        # Secant update formula
        x2 = x1 - df1 * (x1 - x0) / denom

        # Check convergence
        if abs(x2 - x1) < tol:
            return x2, True, k

        # Shift points for next iteration
        x0, df0 = x1, df1
        x1, df1 = x2, df(x2)

    # didn’t converge within maxiter
    return x1, False, maxiter






# Problem 4
def backtracking(f, Df, x, p, alpha=1.0, rho=0.9, c=1e-4):
    """
    Implement the backtracking line search to find a step size that
    satisfies the Armijo condition:

        f(x + α p) ≤ f(x) + c α ∇f(x)ᵀ p.

    Parameters:
        f     (callable): Rⁿ → R objective function.
        Df    (callable): Rⁿ → Rⁿ gradient function ∇f.
        x     (ndarray):  Current point in Rⁿ.
        p     (ndarray):  Search direction in Rⁿ (e.g. negative gradient).
        alpha (float):    Initial step length guess (default 1.0).
        rho   (float):    Reduction factor in (0,1) (default 0.9).
        c     (float):    Armijo constant in (0,1) (default 1e-4).

    Returns:
        alpha (float): A step size satisfying the Armijo sufficient‐decrease condition.
    """
    fx = f(x)
    grad_dot_p = np.dot(Df(x), p)

    # Shrink alpha until the Armijo condition holds
    while f(x + alpha * p) > fx + c * alpha * grad_dot_p:
        alpha *= rho

    return alpha

    
    
    
    
    if __name__ == "__main__":

    # Test f(x) = e^x − 2
    f1  = lambda x: np.exp(x) - 2
    Df1 = lambda x: np.exp(x)
    x0 = 1.0

    x_nm, conv, iters = newton(f1, x0, Df1)
    print("My Newton:", x_nm, "converged?", conv, "iters:", iters)

    x_sp = sp_newton(f1, x0, fprime=Df1, tol=1e-5, maxiter=15)
    print("SciPy Newton:", x_sp)

    # Test f(x) = x^4 − 3
    f2  = lambda x: x**4 - 3
    Df2 = lambda x: 4 * x**3
    x0 = 1.2

    x_nm2, conv2, iters2 = newton(f2, x0, Df2)
    print("My Newton on x^4-3:", x_nm2, "converged?", conv2, "iters:", iters2)
    x_sp2 = sp_newton(f2, x0, fprime=Df2, tol=1e-5, maxiter=15)
    print("SciPy Newton on x^4-3:", x_sp2)
    
    
    
    f  = lambda z: z**3 - 1
    Df = lambda z: 3*z**2

    # Known roots of z^3=1
    zeros = np.array([1, 
                      -0.5 + 0.5j*np.sqrt(3), 
                      -0.5 - 0.5j*np.sqrt(3)])

    # Plot basins
    plot_basins(
        f, Df, zeros,
        domain=[-1.5, 1.5, -1.5, 1.5],
        res=600,    # higher resolution for finer detail
        iters=30    # number of Newton iterations
    )
    
    
    f  = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)

    # Initial guesses
    x0, x1 = 0.0, -1.0
    x_sec, conv, iters = secant1d(df, x0, x1, tol=1e-10, maxiter=500)
    x_sp = newton(df, x0, tol=1e-10, maxiter=500)  # SciPy’s secant

    print(f"Secant1D → x = {x_sec:.12f}, converged? {conv}, iterations = {iters}")
    print(f"SciPy newton (secant) → x = {x_sp:.12f}")

    # Plot f and mark found minimizers
    xs = np.linspace(-2, 4, 800)
    plt.plot(xs, f(xs), label="f(x)")
    plt.axvline(x_sec, color="C1", linestyle="--", label="secant1d minimizer")
    plt.axvline(x_sp,  color="C2", linestyle=":",  label="SciPy newton(min secant)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Secant Method Minimizers vs. f(x)")
    plt.show()
    
    
    
    
    
    
    f  = lambda x: np.dot(x, x)
    Df = lambda x: 2 * x

    x0 = np.array([3.0, 2.0])    # starting point
    p0 = -Df(x0)                 # steepest‐descent direction

    alpha_star = backtracking(f, Df, x0, p0)
    print("Chosen step size α =", alpha_star)

    # Visualize φ(t) = f(x0 + t p0) and mark α*
    ts = np.linspace(0, 1, 200)
    phis = [f(x0 + t * p0) for t in ts]
    plt.plot(ts, phis, label="φ(t)=f(x₀ + t p₀)")
    plt.axvline(alpha_star, color="C1", ls="--", label=f"α*={alpha_star:.3f}")
    plt.xlabel("t")
    plt.ylabel("φ(t)")
    plt.legend()
    plt.title("Backtracking Line Search")
    plt.show()