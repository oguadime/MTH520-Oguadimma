# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Emmanuel Oguadimma
MTH 520
23/05/2025
"""

import numpy as np
import cvxpy as cp

def prob1():
    """Solve the convex optimization problem:

        minimize    2 x1 + x2 + 3 x3
        subject to  x1 + 2 x2       ≤ 3
                    x2 − 4 x3       ≤ 1
                    2 x1 + 10 x2 + 3 x3 ≥ 12
                    x1, x2, x3      ≥ 0

    Returns:
        x_opt (ndarray): the minimizer [x1, x2, x3].
        val   (float):   the optimal objective value.
    """
    # Define variables
    x1 = cp.Variable(nonneg=True)
    x2 = cp.Variable(nonneg=True)
    x3 = cp.Variable(nonneg=True)

    # Objective
    objective = cp.Minimize(2*x1 + x2 + 3*x3)

    # Constraints
    constraints = [
        x1 + 2*x2           <= 3,
        x2 - 4*x3           <= 1,
        2*x1 + 10*x2 + 3*x3 >= 12
        # nonnegativity is enforced by `nonneg=True`
    ]

    # Form and solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract solution
    x_opt = np.array([x1.value, x2.value, x3.value], dtype=float)
    val   = prob.value

    return x_opt, val


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem
         minimize   ‖x‖₁
         subject to Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m,)   ndarray)

    Returns:
        x_opt (ndarray, shape (n,)): the minimizer
        val   (float): the minimal ℓ₁‐norm value
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    # define the n‐vector variable
    x = cp.Variable(n)

    # ℓ₁‐norm objective
    objective = cp.Minimize(cp.norm1(x))

    # equality constraints
    constraints = [A @ x == b]

    # solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract solution
    x_opt = x.value
    val   = prob.value

    return x_opt, val



# Problem 3
def prob3():
    """Solve the piano transportation problem via CVXPY.

    Returns:
        x_opt (ndarray, shape (6,)): optimal shipments [p1…p6].
        cost  (float): total minimal transportation cost.
    """
    # Costs for p1…p6
    c = np.array([4, 7, 6, 8, 8, 9], float)

    # Decision variables p1…p6 ≥ 0
    p = cp.Variable(6, nonneg=True)

    # Supply constraints
    constraints = [
        p[0] + p[1] <= 7,   # center 1 supply
        p[2] + p[3] <= 2,   # center 2 supply
        p[4] + p[5] <= 4,   # center 3 supply
        # Demand constraints (≥ converted directly)
        p[0] + p[2] + p[4] >= 5,  # demand at center 4
        p[1] + p[3] + p[5] >= 8,  # demand at center 5
    ]

    # Objective: minimize total cost cᵀ p
    objective = cp.Minimize(c @ p)

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract solution
    x_opt = p.value
    cost  = prob.value
    return x_opt, cost


# Problem 4
def prob4():
    """Find the minimizer and minimum of
       g(x1,x2,x3) = (3/2)x1^2 + 2 x1 x2 + x1 x3
                  + 2 x2^2 + 2 x2 x3 + (3/2)x3^2
                  + 3 x1 + 1 x3
    Returns:
        x_opt (ndarray, shape (3,)): the minimizer [x1, x2, x3]
        val   (float): the minimal value g(x_opt)
    """
    # Hessian (matrix of second derivatives)
    Q = np.array([[3, 2, 1],
                  [2, 4, 2],
                  [1, 2, 3]], dtype=float)
    # Linear term rᵀ x
    r = np.array([3, 0, 1], dtype=float)

    # Define CVXPY variable
    x = cp.Variable(3)

    # Build the objective: (1/2)xᵀQx + rᵀx
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + r @ x)

    # No constraints (unconstrained convex quadratic)
    prob = cp.Problem(objective)
    prob.solve()

    x_opt = x.value
    val   = prob.value
    return x_opt, val


# Problem 5
def prob5(A, b):
    """Solve
         minimize    ‖A x − b‖₂
         subject to  ‖x‖₁ = 1
                     x ≥ 0

    Parameters:
        A ((m,n) ndarray)
        b ((m,)   ndarray)

    Returns:
        x_opt (ndarray, shape (n,)): the optimal x
        val   (float): the minimal value of ‖A x − b‖₂
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    # decision variable x ∈ ℝⁿ, x ≥ 0
    x = cp.Variable(n, nonneg=True)

    # objective: minimize Euclidean norm of (A x − b)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    # since x ≥ 0, ‖x‖₁ = sum_i x_i
    constraints = [cp.sum(x) == 1]

    # set up and solve
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract solution
    x_opt = x.value
    val   = prob.value
    return x_opt, val


# Problem 6
def prob6(filename="food.npy"):
    """Solve the daily‐food LP:
       minimize    price^T x
       s.t.        (servings * nutrition).T @ x ≥ requirements
                   x ≥ 0

    Assumes food.npy is a pickled dict with keys
      'names', 'price', 'servings', 'nutrition', 'requirements'.
    Returns:
      x_opt (ndarray, length n): optimal packages of each item per day
      cost  (float): minimal daily cost
    """
    # 1) Load the data dict
    raw = np.load(filename, allow_pickle=True)
    data = raw.item()      # unpack the pickled dict

    names        = data["names"]         # list of strings, length n
    price        = np.array(data["price"],      float)  # shape (n,)
    servings     = np.array(data["servings"],   float)  # shape (n,)
    nutrition    = np.array(data["nutrition"],  float)  # shape (n, m)
    requirements = np.array(data["requirements"], float) # shape (m,)

    n, m = nutrition.shape

    # 2) Decision variable: x_i = # packages of food i per day
    x = cp.Variable(n, nonneg=True)

    # 3) Nutrient constraints: for each nutrient j,
    #      sum_i (servings[i] * nutrition[i,j]) * x[i] ≥ requirements[j]
    #    We can form an (n×m) matrix M = servings[:,None] * nutrition
    M = servings[:, None] * nutrition    # shape (n, m)

    constraints = [ M.T @ x >= requirements ]

    # 4) Objective: minimize total cost price^T x
    objective = cp.Minimize(price @ x)

    # 5) Solve with CVXPY
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # 6) Extract solution
    x_opt = x.value           # length-n float array
    cost  = prob.value        # scalar

    return x_opt, cost

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if __name__ == "__main__":
        x_opt, opt_val = prob1()
        print("Optimal x:", x_opt)
        print("Optimal value:", opt_val)
        
        A = np.array([[1, 2,  1,  1],
                  [0, 3, -2, -1]], float)
    b = np.array([7, 4], float)

        x_opt, val = l1Min(A, b)
        print("x_opt =", np.round(x_opt, 6))
        print("l1‐norm =", round(val, 6))
        
        x_opt, total_cost = prob3()
    print("Optimal shipments p1…p6:", x_opt)
    print("Total cost:", total_cost)
    
    x_opt, val = prob4()
    print("Optimizer x =", x_opt)
    print("Minimum value =", val)
    
     # Test using A and b from Problem 2
    A = np.array([[1, 2,  1,  1],
                  [0, 3, -2, -1]], float)
    b = np.array([7, 4], float)

    x_opt, obj_val = prob5(A, b)
    print("Optimizer x:", x_opt)
    print("Objective value:", obj_val)
    
    
    
    x_opt, cost = prob6("food.npy")
    # Which item is eaten most?
    most_idx = int(np.argmax(x_opt))
    print("Most eaten today:",     x_opt[most_idx], "packages of", data["names"][most_idx])
    # And the cost:
    print("Daily cost: $", round(cost,2))

    # If you want to know the top-3 items per week:
    # multiply daily x_opt by 7 and sort descending:
    weekly = 7 * x_opt
    top3 = np.argsort(-weekly)[:3]
    print("Top 3 per week:")
    for i in top3:
        print(f"  {weekly[i]:.1f} of {names[i]}")