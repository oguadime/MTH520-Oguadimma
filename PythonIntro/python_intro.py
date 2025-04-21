# python_intro.py
"""Python Essentials: Introduction to Python.
Emmanuel oguadimma 
 MTH 520
04/12/25
"""


# Problem 2
def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    return 4/3 * 3.14159 * r**3


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, b, c, sep='     ', end=' ')
    print(d,e)

# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    return my_string[:len(my_string)//2]

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]
    


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    # Step 0: Initialize the list
    animals = ["bear", "cat", "dog"]

    # Step 1: Append "eagle"
    animals.append("eagle")

    # Step 2: Replace entry at index 2 with "fox"
    animals[2] = "fox"

    # Step 3: Remove (or pop) entry at index 1
    animals.pop(1)

    # Step 4: Sort the list in reverse alphabetical order
    animals.sort(reverse=True)

    # Step 5: Replace "eagle" with "hawk"
    eagle_index = animals.index("eagle")
    animals[eagle_index] = "hawk"

    # Step 6: Add "hunter" to the last entry
    animals[-1] += "hunter"

    # Return the final list for inspection
    return animals


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = "aeiouAEIOU"
    if word[0] in vowels:
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"


# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    max_pal = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):  # Start from i to avoid duplicate pairs
            product = i * j
            if str(product) == str(product)[::-1] and product > max_pal:
                max_pal = product
    return max_pal

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    return sum([((-1) ** (k + 1)) / k for k in range(1, n + 1)])

    
# Problem 1 (write code below)
if __name__ == "__main__":
    print("Hello, world!") 
    v = sphere_volume(4)
    print(v)
    isolate(1,2,3,4,5)
    print(first_half("Python"))
    print(first_half("Ipython"))
    print(backward("Python"))
    print(backward("Ipython"))
    result = list_ops()
    print(result)
    print(pig_latin("apple"))
    print(pig_latin("banana"))
    print(pig_latin("Orange"))
    print(palindrome())
    print(alt_harmonic(10))   