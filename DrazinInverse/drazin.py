# drazin.py
"""Volume 1: The Drazin Inverse.
Emmanuel Oguadimma 
MTH 520
09/06/2025
"""

import numpy as np
from scipy import linalg as la
from scipy.linalg import schur, inv
from numpy.linalg import matrix_power
import csv


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k, tol=1e-8):
    """
    Verify that Ad is the Drazin inverse of A with index k.

    Conditions:
      1) A @ Ad == Ad @ A
      2) A^(k+1) @ Ad == A^k
      3) Ad @ A @ Ad == Ad

    Parameters:
        A   ((n,n) ndarray): The original matrix.
        Ad  ((n,n) ndarray): Candidate Drazin inverse.
        k   (int):           The index of A (e.g. computed via index(A)).
        tol (float):         Tolerance for equality checks.

    Returns:
        bool: True if all three Drazin conditions hold up to tol.
    """
    # 1) Commutativity
    cond1 = np.allclose(A @ Ad, Ad @ A, atol=tol, rtol=tol)

    # 2) A^(k+1) Ad = A^k
    Ak  = la.matrix_power(A, k)
    Ak1 = la.matrix_power(A, k+1)
    cond2 = np.allclose(Ak1 @ Ad, Ak, atol=tol, rtol=tol)

    # 3) Ad A Ad = Ad
    cond3 = np.allclose(Ad @ A @ Ad, Ad, atol=tol, rtol=tol)

    return cond1 and cond2 and cond3    
    


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """
    Compute the Drazin inverse of A via Algorithm 1 (Schur‐based).

    Parameters:
        A   ((n,n) ndarray): The matrix to invert.
        tol (float):         Tolerance for treating eigenvalues as zero.

    Returns:
        Ad  ((n,n) ndarray): The Drazin inverse of A.
    """
    n = A.shape[0]

    # 1) Schur decomp with |λ|>tol first → Q1, k1
    T1, Q1, k1 = schur(A, sort=lambda x: abs(x) > tol, output='complex')

    # 2) Schur decomp with |λ|≤tol first → Q2, k2
    T2, Q2, k2 = schur(A, sort=lambda x: abs(x) <= tol, output='complex')

    # 3) Build the change‐of‐basis matrix U = [Q1[:, :k1], Q2[:, :n-k1]]
    U = np.hstack((Q1[:, :k1], Q2[:, :n-k1]))
    U_inv = inv(U)

    # 4) Transform A into the block form V = U⁻¹ A U
    V = U_inv @ A @ U

    # 5) Build Z = zeros(n,n), then invert the leading k1×k1 block of V
    Z = np.zeros((n, n), dtype=V.dtype)
    if k1 > 0:
        M = V[:k1, :k1]
        Z[:k1, :k1] = inv(M)

    # 6) Transform back: A^D = U Z U⁻¹
    Ad = U @ Z @ U_inv

    # 7) If A was real, drop tiny imaginary parts
    if np.isrealobj(A):
        Ad = Ad.real

    return Ad

    


# Problem 3
def effective_resistance(A, tol=1e-4):
    """
    Compute the effective resistance between every pair of nodes in an undirected graph.

    Parameters:
        A   ((n,n) ndarray): Adjacency matrix of an undirected graph.
        tol (float):        Tolerance for rounding zero eigenvalues in the Drazin inverse.

    Returns:
        R   ((n,n) ndarray): Matrix where R[i,j] is the effective resistance between
                             node i and node j.
    """
    # 1) Build graph Laplacian L = D − A
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    # 2) Compute the Drazin inverse of L (using your Problem 2 function)
    #    This handles the singularity of L and plays the role of L^† in resistance formulas.
    LD = drazin_inverse(L, tol=tol)

    # 3) Extract the diagonal of LD
    diag_LD = np.diag(LD)

    # 4) Use formula (16.4):
    #       R_ij = L^D_{ii} + L^D_{jj} − L^D_{ij} − L^D_{ji}
    #    Since LD is symmetric, this reduces to
    #       R = diag_LD[:, None] + diag_LD[None, :] − 2*LD
    R = diag_LD[:, None] + diag_LD[None, :] - LD - LD.T

    # 5) Enforce zero on the diagonal (numerical noise)
    np.fill_diagonal(R, 0.0)

    return R
    


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network using effective resistance."""

    def __init__(self, filename='social_network.csv'):
        """Load edge list from CSV, build adjacency matrix, and compute
        the effective resistance matrix.

        Parameters:
            filename (str): Path to a CSV file where each row has two
                            node names connected by an undirected edge.
        """
        # 1. Read edges from the CSV
        edges = []
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    u, v = row[0].strip(), row[1].strip()
                    edges.append((u, v))

        # 2. Extract unique node names in order of first appearance
        names = []
        for u, v in edges:
            if u not in names:
                names.append(u)
            if v not in names:
                names.append(v)
        self.names = names

        # 3. Build adjacency matrix
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        A = np.zeros((n, n), dtype=int)
        for u, v in edges:
            i, j = idx[u], idx[v]
            A[i, j] = 1
            A[j, i] = 1
        self.adj_matrix = A

        # 4. Compute effective resistance matrix (Problem 3)
        #    assumes effective_resistance(A) is defined/imported
        self.eff_resistance = effective_resistance(A)
        


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a particular node."""
        n = len(self.names)
        # Mask out existing edges and self‐loops by setting them to ∞
        mask = np.where(self.adj_matrix > 0, np.inf, self.eff_resistance)
        np.fill_diagonal(mask, np.inf)

        if node is None:
            # Find the global minimum resistance among non‐edges
            i, j = np.unravel_index(np.argmin(mask), mask.shape)
            return self.names[i], self.names[j]
        else:
            # Check that the node exists
            if node not in self.names:
                raise ValueError(f"Node '{node}' not found in network.")
            idx = self.names.index(node)
            # Look only at that row (or column—symmetric)
            col = mask[idx, :]
            # Find the best partner
            j = np.argmin(col)
            return self.names[j]

    def add_link(self, node1, node2):
        """Add a link to the graph between node1 and node2, updating matrices."""
        # Validate nodes
        if node1 not in self.names or node2 not in self.names:
            raise ValueError(f"One or both nodes '{node1}', '{node2}' not in network.")
        i, j = self.names.index(node1), self.names.index(node2)
        # Add the undirected edge
        self.adj_matrix[i, j] = 1
        self.adj_matrix[j, i] = 1
        # Recompute effective resistance
        self.eff_resistance = effective_resistance(self.adj_matrix)
        
                                  
                                  
 
                                  
    if __name__ == "__main__":
    # Example 1
    A  = np.array([[1,3,0,0],
               [0,1,3,0],
               [0,0,1,3],
               [0,0,0,0]], float)
    Ad = np.array([[ 1, -3,  9,  81],
               [ 0,  1, -3, -18],
               [ 0,  0,  1,   3],
               [ 0,  0,  0,   0]], float)
    print(is_drazin(A, Ad, k=1))   

    # Example 2
    B  = np.array([[ 1,  1,  3],
               [ 5,  2,  6],
               [-2, -1, -3]], float)
    Bd = np.zeros((3,3))
    print(is_drazin(B, Bd, k=3)) 
                                  
    # assume index() and is_drazin() are already defined

    A = np.array([[1,3,0,0],
                  [0,1,3,0],
                  [0,0,1,3],
                  [0,0,0,0]], float)
    k = index(A)                       # should return 1
    Ad = drazin_inverse(A, tol=1e-4)
    print("is Drazin?:", is_drazin(A, Ad, k))  # expect True 
                                  
                                  
    A = np.array([[0,1,0,0],
              [1,0,1,0],
              [0,1,0,1],
              [0,0,1,0]], dtype=float)

# Compute the effective resistance matrix
R = effective_resistance(A)

# Print it
print("Effective resistance matrix R:\n", R)  
    
    lp = LinkPredictor('social_network.csv')
print(lp.names)           # list of node names
print(lp.adj_matrix)      # numpy adjacency matrix
print(lp.eff_resistance)  # effective resistance matrix

    lp = LinkPredictor("social_network.csv")

# 1) Global prediction (no argument): returns a tuple of two names
u, v = lp.predict_link()
print(f"The next link should be between {u} and {v}.")

# 2) Per‐node prediction: returns a single name
target = "Alice"
friend = lp.predict_link(target)
print(f"{target} should next connect to {friend}.")

                                  