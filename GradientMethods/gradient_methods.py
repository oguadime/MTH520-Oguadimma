# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Emmanuel Oguadimma 
MTH 520
30/05/2025
"""

import numpy as np
from scipy import linalg as la
from scipy import optimize as opt
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """
    Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f     (callable): Rⁿ → R objective. Accepts an (n,) ndarray and returns float.
        Df    (callable): Rⁿ → Rⁿ gradient ∇f. Accepts and returns an (n,) ndarray.
        x0    ((n,) ndarray): Initial guess.
        tol   (float):   Stopping tolerance on the infinity norm of the gradient.
        maxiter (int):  Maximum number of iterations.

    Returns:
        x     ((n,) ndarray): Approximate minimizer of f.
        converged (bool):     True if ‖∇f(x)‖∞ < tol within maxiter; False otherwise.
        k     (int):          Number of iterations performed.
    """
    x = x0.copy()
    for k in range(1, maxiter + 1):
        grad = Df(x)
        # Check convergence: infinity norm of gradient
        if np.linalg.norm(grad, np.inf) < tol:
            return x, True, k - 1

        # Descent direction is negative gradient
        p = -grad

        # Define phi(alpha) = f(x + alpha * p)
        phi = lambda alpha: f(x + alpha * p)

        # Exact line search: find alpha ≥ 0 minimizing phi
        res = minimize_scalar(phi, bracket=(0, 1), method="Brent")
        alpha_opt = res.x

        # Update x
        x = x + alpha_opt * p

    # If we exit loop, we did not converge within maxiter
    converged = np.linalg.norm(Df(x), np.inf) < tol
    return x, converged, maxiter



# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """
    Solve the linear system Q x = b with the Conjugate Gradient method.

    Parameters:
        Q   ((n,n) ndarray): Symmetric positive‐definite matrix.
        b   ((n,)   ndarray): Right‐hand side vector.
        x0  ((n,)   ndarray): Initial guess for the solution.
        tol (float):         Convergence tolerance on the residual norm.

    Returns:
        x          ((n,) ndarray): Approximate solution to Q x = b.
        converged  (bool):         True if ‖r_k‖ < tol within at most n steps.
        k          (int):          Number of iterations performed (≤ n).
    """
    x = x0.copy()
    r = b - Q @ x            # initial residual
    p = r.copy()             # initial search direction
    rr_old = np.dot(r, r)

    if np.sqrt(rr_old) < tol:
        return x, True, 0

    n = Q.shape[0]
    for k in range(1, n+1):
        Qp = Q @ p
        alpha = rr_old / np.dot(p, Qp)

        # Update approximate solution
        x = x + alpha * p

        # Update residual
        r = r - alpha * Qp
        rr_new = np.dot(r, r)

        # Check convergence
        if np.sqrt(rr_new) < tol:
            return x, True, k

        # Compute next direction coefficient
        beta = rr_new / rr_old

        # Update search direction
        p = r + beta * p

        rr_old = rr_new

    # If reached here, did not converge within n iterations
    return x, False, n






# Problem 3
Q = np.array([[2.0, 0.0],
                  [0.0, 4.0]])
    b = np.array([1.0, 8.0])
    x0 = np.zeros(2)

    x_cg, converged, iters = conjugate_gradient(Q, b, x0, tol=1e-6)
    x_true = la.solve(Q, b)

    print("Conjugate Gradient solution:", x_cg)
    print("Exact solution (numpy.linalg.solve):", x_true)
    print("Converged?", converged, "in iterations:", iters)

    # Test on a random SPD system:
    np.random.seed(0)
    n = 5
    A = np.random.randn(n, n)
    Q2 = A.T @ A + 1e-3 * np.eye(n)  # ensure positive‐definite
    b2 = np.random.randn(n)
    x0_2 = np.zeros(n)

    x_cg2, conv2, iters2 = conjugate_gradient(Q2, b2, x0_2, tol=1e-8)
    x_true2 = la.solve(Q2, b2)

    print("\nRandom SPD test:")
    print("CG solution:", x_cg2)
    print("Exact solution:", x_true2)
    print("Norm of (CG − exact):", np.linalg.norm(x_cg2 - x_true2))
    print("Converged?", conv2, "in iterations:", iters2)


# Problem 4
def nonlinear_conjugate_gradient(f, Df, x0, tol=1e-5, maxiter=100):
    """
    Compute the minimizer of f using the nonlinear conjugate gradient method
    (Fletcher–Reeves version).

    Parameters:
        f     (callable): Rⁿ → R objective function. Accepts an (n,) ndarray, returns float.
        Df    (callable): Rⁿ → Rⁿ gradient ∇f. Accepts and returns an (n,) ndarray.
        x0    ((n,) ndarray): Initial guess.
        tol   (float): Convergence tolerance on the norm of the gradient.
        maxiter (int): Maximum number of iterations.

    Returns:
        x        ((n,) ndarray): Approximate minimizer of f.
        converged (bool):       True if ‖∇f(x)‖ < tol within maxiter; False otherwise.
        k        (int):         Number of iterations performed.
    """
    x = x0.copy()
    g = Df(x)
    d = -g                         # initial search direction
    g_norm = np.linalg.norm(g)

    if g_norm < tol:
        return x, True, 0

    for k in range(1, maxiter + 1):
        # Define phi(alpha) = f(x + alpha * d)
        phi = lambda alpha: f(x + alpha * d)

        # Perform a line search to find alpha ≥ 0 minimizing phi
        res = minimize_scalar(phi, bracket=(0, 1), method="Brent")
        alpha = res.x

        # Update x
        x_new = x + alpha * d
        g_new = Df(x_new)

        # Check convergence
        if np.linalg.norm(g_new) < tol:
            return x_new, True, k

        # Fletcher–Reeves beta
        beta = np.dot(g_new, g_new) / np.dot(g, g)

        # Update direction
        d = -g_new + beta * d

        # Prepare for next iteration
        x, g = x_new, g_new

    # If we reach here, did not converge within maxiter
    return x, False, maxiter


    
    


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """
        Choose the optimal parameters beta0 and beta1 by minimizing the negative
        log-likelihood of the logistic model.

        Parameters:
            x     ((n,) ndarray):        Predictor variables.
            y     ((n,) ndarray of 0/1): Outcome labels.
            guess ((2,) array-like):     Initial guess for [beta0, beta1].
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")

        def sigmoid(z):
            # Numerically stable sigmoid
            return 1.0 / (1.0 + np.exp(-z))

        def negative_log_likelihood(beta):
            """
            Negative log-likelihood for 1D logistic regression:
              beta = [beta0, beta1]
              eta_i = beta0 + beta1 * x_i
              p_i = sigmoid(eta_i)
              loglik = sum[ y_i*log(p_i) + (1 - y_i)*log(1 - p_i) ]
              return -loglik
            """
            beta0, beta1 = beta
            eta = beta0 + beta1 * x
            p = sigmoid(eta)
            # clip p to avoid log(0)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Use scipy.optimize.fmin_cg to minimize the negative log-likelihood
        beta_opt = fmin_cg(negative_log_likelihood, x0=np.asarray(guess), disp=False)

        # Store fitted parameters as attributes
        self.beta0, self.beta1 = beta_opt

    def predict(self, x):
        """
        Given a new predictor x (float or array-like), compute
            p = 1 / (1 + exp(-(beta0 + beta1 * x)))
        where beta0 and beta1 were determined in fit().

        Parameters:
            x (float or array-like): Input predictor(s).

        Returns:
            float or ndarray: Probability P(y=1 | x).
        """
        # Ensure fit() has been called
        if not hasattr(self, "beta0") or not hasattr(self, "beta1"):
            raise AttributeError("Must call fit() before predict()")

        x_arr = np.asarray(x)
        eta = self.beta0 + self.beta1 * x_arr
        return 1.0 / (1.0 + np.exp(-eta))




# Problem 6
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """
        Choose the optimal parameters beta0 and beta1 by minimizing the negative
        log‐likelihood of the logistic model.

        Parameters:
            x     ((n,) ndarray):        Predictor variables.
            y     ((n,) ndarray of 0/1): Outcome labels.
            guess ((2,) array‐like):     Initial guess for [beta0, beta1].
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        def negative_log_likelihood(beta):
            beta0, beta1 = beta
            eta = beta0 + beta1 * x
            p = sigmoid(eta)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        beta_opt = fmin_cg(negative_log_likelihood, x0=np.asarray(guess), disp=False)
        self.beta0, self.beta1 = beta_opt

    def predict(self, x):
        """
        Given a new predictor x (float or array‐like), compute
            p = 1 / (1 + exp(−(beta0 + beta1 * x))).
        """
        if not hasattr(self, "beta0") or not hasattr(self, "beta1"):
            raise AttributeError("Must call fit() before predict()")
        x_arr = np.asarray(x)
        eta = self.beta0 + self.beta1 * x_arr
        return 1.0 / (1.0 + np.exp(-eta))


def prob6(filename="challenger.npy", guess=np.array([20.0, -1.0])):
    """
    Return the probability of O‐ring damage at 31°F and plot the fitted logistic
    curve on [30, 100] against the raw Challenger data.

    Parameters:
        filename (str): The .npy file containing Challenger data. Defaults to "challenger.npy".
                        Each row has [temperature, damage_flag].
        guess    ((2,) ndarray): Initial guess for [beta0, beta1]. Defaults to [20., -1.].

    Returns:
        float: P(damage = 1 | temperature = 31°F) according to the fitted model.
    """
    # Load the Challenger dataset: each row = [temperature (°F), damage_flag (0 or 1)]
    data = np.load(filename)
    temps = data[:, 0]
    damage = data[:, 1]

    # Fit a 1D logistic regression model
    model = LogisticRegression1D()
    model.fit(temps, damage, guess)

    # Generate a smooth curve from 30°F to 100°F
    x_curve = np.linspace(30, 100, 300)
    y_curve = model.predict(x_curve)

    # Plot raw data (blue dots at y=0 or y=1) and fitted curve (orange line)
    plt.figure(figsize=(8, 5))
    plt.scatter(temps, damage, color="tab:blue", alpha=0.6, label="Observed Damage (0/1)")
    plt.plot(x_curve, y_curve, color="tab:orange", linewidth=2, label="Fitted P(Damage | temp)")

    plt.xlabel("Temperature (°F)")
    plt.ylabel("O‐Ring Damage Probability")
    plt.title("Probability of O‐Ring Damage vs. Launch Temperature")
    plt.ylim([-0.05, 1.05])
    plt.xlim([30, 100])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute and return the predicted probability at 31°F
    p_at_31 = model.predict(31.0)
    return p_at_31

    
    
    
    
    
    
    
    if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) Test on f(x,y,z) = x^4 + y^4 + z^4
    f1 = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    Df1 = lambda x: np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])

    x0 = np.array([1.0, -1.5, 2.0])
    x_min1, conv1, iters1 = steepest_descent(f1, Df1, x0, tol=1e-8, maxiter=500)
    print("f1 minimizer:", x_min1)
    print("Converged?", conv1, "in iterations:", iters1)
    print("f1(x_min):", f1(x_min1))

    # 2) Test on the Rosenbrock function f(x,y) = 100(x2 - x1^2)^2 + (1 - x1)^2
    def f2(x):
        return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    def Df2(x):
        # ∂f/∂x1 = -400 x1 (x2 - x1^2) - 2(1 - x1)
        # ∂f/∂x2 = 200 (x2 - x1^2)
        return np.array([
            -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0]),
             200.0 * (x[1] - x[0]**2)
        ])

    x0_rosen = np.array([-1.2, 1.0])
    x_min2, conv2, iters2 = steepest_descent(f2, Df2, x0_rosen, tol=1e-6, maxiter=2000)
    print("\nRosenbrock minimizer:", x_min2)
    print("Converged?", conv2, "in iterations:", iters2)
    print("f2(x_min):", f2(x_min2))

    # Plot f2 along the search trajectory (optional)
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 100.0 * (Y - X**2)**2 + (1 - X)**2

    plt.figure(figsize=(6,5))
    # contour plot of Rosenbrock
    levels = np.logspace(-1, 3, 20)
    plt.contour(X, Y, Z, levels=levels, norm=plt.LogNorm(), cmap="viridis")
    plt.plot(x_min2[0], x_min2[1], 'r*', markersize=12, label="Found minimizer")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Rosenbrock function contours and found minimizer")
    plt.legend()
    plt.show()
    
    
    # Rosenbrock function in R²: f(x,y) = 100(y − x²)² + (1 − x)²
    f_rosen = lambda v: 100.0 * (v[1] - v[0]**2)**2 + (1.0 - v[0])**2
    # Gradient:
    #   ∂f/∂x = −400 x (y − x²) − 2(1 − x)
    #   ∂f/∂y =  200 (y − x²)
    Df_rosen = lambda v: np.array([
        -400.0 * v[0] * (v[1] - v[0]**2) - 2.0 * (1.0 - v[0]),
         200.0 * (v[1] - v[0]**2)
    ])

    # Initial guess
    x0 = np.array([10.0, 10.0])

    x_min, converged, iters = nonlinear_conjugate_gradient(f_rosen, Df_rosen,
                                                           x0, tol=1e-6, maxiter=500)
    print("NCG found minimizer:", x_min)
    print("Converged?", converged, "in iterations:", iters)
    print("f(x_min) =", f_rosen(x_min))

    # Compare to SciPy’s built‐in:
    from scipy.optimize import fmin_cg
    x_sp = fmin_cg(f_rosen, x0, fprime=Df_rosen, gtol=1e-6, maxiter=500, disp=False)
    print("SciPy fmin_cg location:", x_sp)
    print("SciPy fmin_cg f-value:", f_rosen(x_sp))

    # Optional: contour plot showing the solution
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1,  3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 100.0 * (Y - X**2)**2 + (1 - X)**2

    plt.figure(figsize=(6,5))
    levels = np.logspace(-1, 3, 20)
    plt.contour(X, Y, Z, levels=levels, norm=plt.LogNorm(), cmap="viridis")
    plt.plot(x_min[0], x_min[1], 'r*', markersize=12, label="NCG minimizer")
    plt.plot(x_sp[0],  x_sp[1],  'bx', markersize=8, label="SciPy fmin_cg")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Rosenbrock contours and NCG result")
    plt.legend()
    plt.show()
    
    
    
    np.random.seed(0)
    n_samples = 100
    x_data = np.linspace(-3, 3, n_samples)
    # True parameters: beta0 = -0.5, beta1 = 1.2
    beta0_true, beta1_true = -0.5, 1.2
    eta = beta0_true + beta1_true * x_data
    probs = 1.0 / (1.0 + np.exp(-eta))
    # Draw labels y_data from Bernoulli(probs)
    y_data = np.random.binomial(1, probs)

    # Fit the logistic regressor
    model = LogisticRegression1D()
    initial_guess = [0.0, 0.0]   # beta0 = 0, beta1 = 0
    model.fit(x_data, y_data, initial_guess)

    print(f"Fitted beta0 = {model.beta0:.4f}, beta1 = {model.beta1:.4f}")
    print(f"True    beta0 = {beta0_true:.4f}, beta1 = {beta1_true:.4f}")

    # Plot the data and the fitted sigmoid curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    # scatter: jitter y by a small amount for visibility
    jitter = (np.random.rand(n_samples) - 0.5) * 0.05
    plt.scatter(x_data, y_data + jitter, s=20, alpha=0.6, label="data")

    # draw fitted probability curve
    x_plot = np.linspace(-3, 3, 300)
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label="fitted sigmoid")

    plt.xlabel("x")
    plt.ylabel("P(y=1 | x)")
    plt.title("1D Logistic Regression Fit")
    plt.legend()
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.show()

    # Predict on a new point
    x_new = 0.7
    prob_new = model.predict(x_new)
    print(f"Predicted P(y=1) at x={x_new:.2f} is {prob_new:.4f}")
    
    
    
    probability_31 = prob6("challenger.npy", guess=np.array([20.0, -1.0]))
    print(f"Predicted probability of O-ring damage at 31°F: {probability_31:.4f}")
    
    
    
    