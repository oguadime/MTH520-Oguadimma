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
    raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    raise NotImplementedError("Problem 4 Incomplete")


def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    raise NotImplementedError("Problem 5 Incomplete")


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
    raise NotImplementedError("Problem 6 Incomplete")


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    raise NotImplementedError("Problem 7 Incomplete")

    
if __name__ == "__main__":
    print("A @ B =\n", prob1()) 
    print("Result of -A^3 + 9A^2 - 15A =")
    print(prob2())