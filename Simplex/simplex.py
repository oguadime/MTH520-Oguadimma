"""Volume 2: Simplex

Emmanuel Oguadimma 
MTH 520
16/06/2025
"""

import numpy as np

# Problems 1-6
class SimplexSolver:
    """Class for solving the standard linear optimization problem

                    minimize   c^T x
                    subject to A x ≤ b,   x ≥ 0
    via the Simplex algorithm.
    """

    def __init__(self, c, A, b):
        """Check feasibility at the origin (x = 0) and store the problem data.

        Parameters:
            c ((n,) ndarray): Objective coefficients.
            A ((m,n) ndarray): Constraint matrix.
            b ((m,) ndarray): Right‐hand side vector.

        Raises:
            ValueError: If b has any negative entry (i.e. A·0 = 0 ≰ b).
        """
        # Convert inputs to arrays
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)

        # Shape checks
        m, n = self.A.shape
        if self.c.shape != (n,):
            raise ValueError(f"c must have shape ({n},), got {self.c.shape}")
        if self.b.shape != (m,):
            raise ValueError(f"b must have shape ({m},), got {self.b.shape}")

        # Feasibility at the origin: A·0 = 0 must satisfy 0 ≤ b elementwise
        if np.any(self.b < 0):
            neg_idx = np.where(self.b < 0)[0]
            raise ValueError(
                f"Infeasible at origin: b[{neg_idx.tolist()}] < 0"
            )

        # At this point, the system is feasible at x = 0.
        # You can now initialize your simplex tableau or dictionary as needed.
        
        

    # Problem 2
    def __init__(self, c, A, b):
        # … feasibility check as before …
        # now build the dictionary
        self._generate_dictionary(self.c, self.A, self.b)

    def _generate_dictionary(self, c, A, b):
        """Generate the initial simplex dictionary in NumPy form.

        We have:
            minimize   cᵀ x
            subject to A x ≤ b,  x ≥ 0

        → introduce slack s ≥ 0 so that A x + I s = b

        The dictionary has rows
          s_i = b_i − ∑_j A[i,j] x_j
        and
          z   = 0 + ∑_j c[j] x_j

        Parameters:
            c ((n,) ndarray): Objective coefficients.
            A ((m,n) ndarray): Constraint matrix.
            b ((m,) ndarray): Right‐hand side.
        """
        m, n = A.shape
        # (m+1) rows:  m constraints + 1 objective
        # (1 + n + m) columns: RHS | x_1…x_n | s_1…s_m
        D = np.zeros((m+1, 1 + n + m), dtype=float)

        # 1) Constraint rows: s_i = b_i − ∑ A[i,j] x_j
        D[:m, 0]       = b                     # RHS = b
        D[:m, 1:1+n]   = -A                    # move A x to RHS
        D[:m, 1+n:1+n+m] = np.eye(m)           # slack s_i coefficient

        # 2) Objective row: z = 0 + ∑ c[j] x_j
        D[m, 0]        = 0                     # objective value
        D[m, 1:1+n]    =  c                    # c_j in front of x_j
        # slack vars don’t appear in objective → zeros in D[m,1+n:]

        # store on the instance
        self.dictionary = D


    # Problem 3a
        def _pivot_col(self):
        """Return the column index of the entering variable according to Bland’s Rule:
        choose the smallest‐index column with a negative cost in the objective row."""
        # objective row is the last row of the dictionary
        obj = self.dictionary[-1, :]
        n_cols = obj.size
        # scan columns 1…end (skip column 0 which is RHS)
        for j in range(1, n_cols):
            if obj[j] < 0:
                return j
        # optimal if no negative cost
        return None

    def _pivot_row(self, pivot_col):
        """Return the row index of the leaving variable via the ratio test
        and Bland’s Rule.  Among rows with positive coeff in pivot_col,
        choose the smallest RHS/coeff ratio; break ties by smallest row index."""
        D = self.dictionary
        m, _ = D.shape
        m -= 1  # last row is objective, so there are m constraint rows
        best_ratio = np.inf
        best_row = None

        for i in range(m):
            a_ij = D[i, pivot_col]
            if a_ij > 0:
                ratio = D[i, 0] / a_ij
                # choose strictly smaller ratio, or tie by smaller row index
                if (ratio < best_ratio - 1e-12) or (abs(ratio - best_ratio) < 1e-12 and (best_row is None or i < best_row)):
                    best_ratio = ratio
                    best_row = i

        return best_row

        

    # Problem 4
       def pivot(self):
        """Perform one Bland’s‐Rule pivot (entering + leaving) on the current dictionary.
        Raises ValueError if the problem is unbounded."""
        # 1) Choose entering column
        j = self._pivot_col()
        if j is None:
            # No negative cost ⇒ already optimal ⇒ nothing to do
            return

        # 2) Choose leaving row
        i = self._pivot_row(j)
        if i is None:
            # No positive entries in column j ⇒ unbounded
            raise ValueError("Linear program is unbounded (no valid pivot row).")

        # 3) Pivot on (i,j): make column j into the i-th unit vector
        D = self.dictionary
        pivot_val = D[i, j]

        # a) Scale pivot row to make pivot element = 1
        D[i, :] = D[i, :] / pivot_val

        # b) Eliminate j‐th column in all other rows
        m, _ = D.shape
        for r in range(m):
            if r == i:
                continue
            factor = D[r, j]
            D[r, :] = D[r, :] - factor * D[i, :]

        # 4) Store updated dictionary
        self.dictionary = D

        

    # Problem 5
        def solve(self):
        """Run the Simplex algorithm to optimality and return
        (min_value, basic_vars, nonbasic_vars).

        Returns:
            float: the minimum objective value.
            dict: basic_vars mapping var_index→value.
            dict: nonbasic_vars mapping var_index→value.
        Raises:
            ValueError: if the LP is unbounded.
        """
        # 1) Perform pivots until no negative cost remains
        while True:
            j = self._pivot_col()
            if j is None:
                break  # optimal
            i = self._pivot_row(j)
            if i is None:
                raise ValueError("LP is unbounded.")
            self.pivot()

        D = self.dictionary
        m, total_cols = D.shape[0] - 1, D.shape[1]
        # total_vars = total_cols - 1  # exclude RHS

        basic_vars = {}
        nonbasic_vars = {}

        # 2) Identify basic vs. nonbasic by looking for unit columns in constraint rows
        for col in range(1, total_cols):
            column = D[:m, col]
            # check if it's a unit vector
            ones = np.isclose(column, 1.0)
            zeros = np.isclose(column, 0.0)
            if ones.sum() == 1 and zeros.sum() == (m - 1):
                # basic: find the row where it's 1
                row = int(np.where(ones)[0][0])
                var_index = col - 1
                basic_vars[var_index] = float(D[row, 0])
            else:
                # nonbasic: value = 0
                var_index = col - 1
                nonbasic_vars[var_index] = 0.0

        # 3) The objective value is in the last row, first column
        min_value = float(D[m, 0])

        return min_value, basic_vars, nonbasic_vars
    
        def test_simplex_example():
        c = np.array([-3, -2])
        A = np.array([[1, -1],
                  [3,  1],
                  [4,  3]])
        b = np.array([2, 5, 7])

        solver = SimplexSolver(c, A, b)
        val, basic, nonbasic = solver.solve()

        assert abs(val + 5.2) < 1e-8,      "Objective value incorrect"
        # Basic variables should be x0=1.6,x1=0.2,s1=0.6
        expected_basic = {0: 1.6, 1: 0.2, 2: 0.6}
        assert basic == expected_basic
        # Nonbasic (slacks 2 and 3 in this problem) should be zero
        expected_nonbasic = {3: 0.0, 4: 0.0}
        assert nonbasic == expected_nonbasic

       

    # Problem 6
    def prob6(filename='productMix.npz'):
    """Solve the product‐mix LP from productMix.npz via your SimplexSolver.

    Returns:
        x_opt ((n,) ndarray): Optimal production quantities for each product.
    """
    # 1) Load the data
    data = np.load(filename)
    A_res = data['A']   # shape (num_resources, num_products)
    p     = data['p']   # unit prices, length = num_products
    m     = data['m']   # available units of each resource, length = num_resources
    d     = data['d']   # demand bounds for each product, length = num_products

    # 2) Build the standard‐form LP:
    #     maximize p^T x
    #     s.t.  A_res @ x ≤ m      (resource constraints)
    #           I @ x     ≤ d      (demand constraints)
    #           x ≥ 0
    #
    # Our solver does minimization, so set c = –p.
    c = -p

    # Stack the two sets of constraints
    num_res, num_prod = A_res.shape
    I_demand = np.eye(num_prod)
    A_stack = np.vstack([A_res, I_demand])           # shape (num_res+num_prod, num_prod)
    b_stack = np.concatenate([m, d])                 # length num_res+num_prod

    # 3) Solve with your SimplexSolver
    solver = SimplexSolver(c, A_stack, b_stack)
    _, basic_vars, nonbasic_vars = solver.solve()

    # 4) Extract the optimal x_i for i=0…num_prod−1
    x_opt = np.zeros(num_prod, dtype=float)
    for i in range(num_prod):
        # variable indices 0…num_prod−1 correspond to x_i
        x_opt[i] = basic_vars.get(i, nonbasic_vars.get(i, 0.0))

    return x_opt

    
    
    
    
    
    if __name__ == "__main__":
    # Example LP:
    #   minimize    x + y
    #   subject to  x + 2y ≤ 4
    #                x +  y ≤ 2
    #                x, y ≥ 0
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 2.0],
                  [1.0, 1.0]])
    b = np.array([4.0, 2.0])

    # This will check feasibility at the origin and build the solver
    try:
        solver = SimplexSolver(c, A, b)
        print("Origin is feasible. SimplexSolver initialized successfully.")
        # You could now call solver.solve() or whatever method you implement next.
    except ValueError as e:
        print("Feasibility check failed:", e)
      
    # 2 Define a small LP to test on:
    #    minimize   x + y
    #    subject to x + 2y ≤ 4
    #               x +  y ≤ 2
    #               x,y ≥ 0
    c = np.array([1.0, 1.0])
    A = np.array([[1.0, 2.0],
                  [1.0, 1.0]])
    b = np.array([4.0, 2.0])
    solver = SimplexSolver(c,A,b)
    print(solver.dictionary)
        
    # 3 Instantiate your solver (this builds self.dictionary)
    solver = SimplexSolver(c, A, b)

    # Print the initial dictionary
    print("Initial dictionary:\n", solver.dictionary)

    # Pick the pivot column (entering variable)
    j = solver._pivot_col()
    print("Pivot column index:", j)

    # Pick the pivot row (leaving variable)
    i = solver._pivot_row(j)
    print("Pivot row index:", i)
    
    # 4 solver.pivot()
    print(solver.dictionary)
    
    # 5 Example LP from your test:
    c = np.array([-3, -2])
    A = np.array([[1, -1],
                  [3,  1],
                  [4,  3]])
    b = np.array([2, 5, 7])

    solver = SimplexSolver(c, A, b)
    value, basic, nonbasic = solver.solve()

    print("Optimal value:", value)
    print("Basic variables:", basic)
    print("Nonbasic variables:", nonbasic)
    
    # 6 x_opt = prob6("productMix.npz")
    print(x_opt)