# standard_library.py
"""Python Essentials: The Standard Library.
Emmanuel oguadimma 
 MTH 520
04/18/25
"""

from math import sqrt
import calculator
import time
from box import isvalid, parse_input


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    return min(L), max(L), sum(L) / len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    # Integer test
    a = 1
    b = a
    a += 1
    int_same = (a == b)

    # String test
    s1 = "hello"
    s2 = s1
    s1 += " world"
    str_same = (s1 == s2)

    # List test
    l1 = [1, 2, 3]
    l2 = l1
    l1.append(4)
    list_same = (l1 == l2)

    # Tuple test
    t1 = (1, 2, 3)
    t2 = t1
    t1 += (4,)
    tuple_same = (t1 == t2)

    # Set test
    set1 = {1, 2, 3}
    set2 = set1
    set1.add(4)
    set_same = (set1 == set2)
    
    # Print results
    print("int:   a == b        ->", int_same)
    print("str:   s1 == s2      ->", str_same)
    print("list:  l1 == l2      ->", list_same)
    print("tuple: t1 == t2      ->", tuple_same)
    print("set:   set1 == set2  ->", set_same)

    # Return a dict of results
    return {
        'int':    int_same,
        'str':    str_same,
        'list':   list_same,
        'tuple':  tuple_same,
        'set':    set_same,
    }
    


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    sum_of_squares = calculator.sum(
        calculator.product(a, a),
        calculator.product(b, b)
    )
    # √(a² + b²)
    return calculator.sqrt(sum_of_squares)
    


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    items = list(A)
    return [
        set(combo)
        for r in range(len(items) + 1)
        for combo in combinations(items, r)
    ]
    


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    print(f"Welcome, {player}! You have {timelimit} seconds to shut the box.")
    remaining = list(range(1, 10))
    start = time.time()

    while remaining and (time.time() - start) < timelimit:
        roll = ...                # your dice‑rolling logic here
        if not isvalid(roll, remaining):
            print("No valid moves—game over!")
            break

        print(f"Roll: {roll}  Remaining: {remaining}")
        choice = input("Which numbers to knock down? ")
        picks = parse_input(choice, remaining)
        if not picks:
            print("Invalid choice, try again.")
            continue

        for num in picks:
            remaining.remove(num)

    score = sum(remaining)
    print(f"Time’s up! Your score is {score}.")
    return score

    

    
    
if __name__ == "__main__":
    low, high, avg = prob1([3, 7, 2, 9, 4])
    print(low, high, avg)
    results = prob2()
    hypotenuse(3, 4)
    example = { 'a', 'b', 'c' }
    ps = power_set(example)
    print(ps)
    
    import sys
    if len(sys.argv) != 3:
        print("Usage: python shut_the_box.py <player_name> <time_limit_seconds>")
        sys.exit(1)

    player_name = sys.argv[1]
    try:
        time_limit = int(sys.argv[2])
    except ValueError:
        print("Time limit must be an integer number of seconds.")
        sys.exit(1)

    shut_the_box(player_name, time_limit)