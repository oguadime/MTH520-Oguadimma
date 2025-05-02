# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Emmanuel oguadimma 
 MTH 520
04/18/25
"""

import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    A = np.array([
        [3, -1, 4],
        [1, 5, -9]
    ])
    B = np.array([
        [2, 6, -5, 3],
        [5, -8, 9, 7], 
        [9, -3, -2, -3]
    ])
    # Matrix‐multiply A and B and return the result
    return A @ B


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    A = np.array([
        [3, 1, 4],
        [1, 5, 9],
        [-5, 3, 1]
    ])

    # Compute powers of A
    A2 = A @ A         # A squared
    A3 = A2 @ A        # A cubed

    # Evaluate the polynomial ‑A^3 + 9A^2 ‑ 15A
    return -A3 + 9 * A2 - 15 * A


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    N = 7

    # A = upper‐triangular 7×7 of ones
    A = np.triu(np.ones((N, N), dtype=int), k=0)

    # B[i,j] = –1 for j ≤ i, +5 for j > i
    # start with all 5's, then subtract 6 on & below the diagonal (5 - 6 = -1)
    B = 5 * np.ones((N, N), dtype=int) - 6 * np.tri(N, N, k=0, dtype=int)

    # compute ABA and cast to int64
    C = A @ B @ A
    return C.astype(np.int64)



def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = A.copy()
    # set all negative entries to zero using fancy indexing
    B[B < 0] = 0
    return B
   


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    # define the small blocks
    A = np.array([[0, 2, 4],
                  [1, 3, 5]], dtype=int)           # 2×3
    B = np.array([[3, 0, 0],
                  [3, 3, 0],
                  [3, 3, 3]], dtype=int)           # 3×3
    C = np.diag([-2, -2, -2], dtype=int)           # 3×3 diagonal of –2’s
    I3 = np.eye(3, dtype=int)                      # 3×3 identity

    # zero‐blocks of appropriate shapes
    Z33 = np.zeros((3, 3), dtype=int)
    Z32 = np.zeros((3, 2), dtype=int)
    Z23 = np.zeros((2, 3), dtype=int)
    Z22 = np.zeros((2, 2), dtype=int)

    # first block‐row: [ 0₃ₓ₃ , Aᵀ , I₃ ]
    top    = np.hstack([Z33,    A.T,  I3])

    # second block‐row: [ A , 0₂ₓ₂ , 0₂ₓ₃ ]
    middle = np.hstack([A,      Z22,  Z23])

    # third block‐row: [ B , 0₃ₓ₂ , C   ]
    bottom = np.hstack([B,      Z32,  C  ])

    # stack the three rows
    M = np.vstack([top, middle, bottom])
    return M
    


def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    # compute row sums as a column vector (n×1)
    row_sums = A.sum(axis=1, keepdims=True)
    # broadcast division across each row
    return A / row_sums
    


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    # horizontal products of length 4
    horiz = grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:]
    # vertical products of length 4
    vert  = grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]
    # diagonal down-right
    diag_dr = grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]
    # diagonal up-right
    diag_ur = grid[3:, :-3]  * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:]

    # find the maximum among all
    max_prod = max(horiz.max(), vert.max(), diag_dr.max(), diag_ur.max())
    return int(max_prod)
    

    
if __name__ == "__main__":
    print("A @ B =\n", prob1()) 
    print("Result of -A^3 + 9A^2 - 15A =")
    print(prob2())
    print(prob3())
    A = np.array([-3, -1, 3])
    print(prob4(A))
    print(prob5())
    A = np.array([[1,1,0],[0,1,0],[1,1,1]])
    print(prob6(A))
    print(prob7()) 