# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Emmanuel Oguadimma
MTH 520
04/25/25
"""

from math import sqrt


class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    class Backpack:
    """A backpack that holds items up to a fixed capacity.

    Attributes:
        name (str): The owner’s name.
        color (str): The backpack’s color.
        max_size (int): Maximum number of items the backpack can hold.
        contents (list): The list of items currently in the backpack.
    """

    def __init__(self, name, color, max_size=5):
        """Initialize a new Backpack.

        Parameters:
            name (str): The name of the backpack’s owner.
            color (str): The color of the backpack.
            max_size (int, optional): Capacity (default is 5).
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack if there’s room.

        If the backpack is already at or above capacity, prints “No Room!” and
        does not add the item.

        Parameters:
            item: Anything to store in the backpack.
        """
        if len(self.contents) >= self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack’s contents.

        Parameters:
            item: The item to remove (must already be in contents).
        """
        self.contents.remove(item)

    def dump(self):
        """Empty all contents from the backpack."""
        self.contents = []


def test_backpack():
    """Simple tests for the Backpack class."""
    testpack = Backpack("Barry", "black")   # Instantiate the object.

    # Test attributes
    if testpack.name != "Barry":
        print("Backpack.name assigned incorrectly")
    if testpack.color != "black":
        print("Backpack.color assigned incorrectly")
    if testpack.max_size != 5:
        print("Backpack.max_size default assigned incorrectly")

    # Test put() up to capacity
    for item in ["pencil", "pen", "paper", "computer", "book", "snack"]:
        testpack.put(item)
        print("Contents:", testpack.contents)

    # At this point, only the first 5 items should be in contents, and “No Room!”
    # should have been printed once for “snack”.

    # Test dump()
    testpack.dump()
    if testpack.contents:
        print("dump() failed to empty contents")
    else:
        print("dump() succeeded; contents now empty")

    print("Final contents:", testpack.contents)


    # Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
    class Jetpack(Backpack):
    """A Jetpack is a Backpack with a built-in fuel tank that allows “flight.”

    Inherits all attributes and methods of Backpack (name, color, max_size, contents,
    put, take), but limits capacity by default and adds fuel management and flight.

    Attributes:
        name (str): The owner’s name (inherited).
        color (str): The jetpack’s color (inherited).
        max_size (int): Capacity for carried items (default 2).
        contents (list): Items currently stored (inherited).
        fuel (float): Current amount of fuel in the tank.
    """

    def __init__(self, name, color, max_size=2, fuel=10):
        """Initialize a new Jetpack.

        Parameters:
            name (str): Owner’s name.
            color (str): Jetpack color.
            max_size (int, optional): Maximum number of items it can carry. Defaults to 2.
            fuel (float, optional): Initial fuel level. Defaults to 10.
        """
        super().__init__(name, color, max_size=max_size)
        self.fuel = fuel

    def fly(self, amount):
        """Attempt to burn a specified amount of fuel to fly.

        Parameters:
            amount (float): Amount of fuel to consume.

        Behavior:
            - If there is enough fuel (self.fuel >= amount), subtracts it from self.fuel.
            - Otherwise, prints “Not enough fuel!” and leaves fuel unchanged.
        """
        if amount <= self.fuel:
            self.fuel -= amount
        else:
            print("Not enough fuel!")

    def dump(self):
        """Empty both the contents of the backpack and the fuel tank."""
        # Empty contents list (inherited)
        self.contents = []
        # Empty fuel tank
        self.fuel = 0



    # Magic Methods -----------------------------------------------------------

    def __eq__(self, other):
        """Two backpacks are equal iff they have the same name, color, and number of items."""
        if not isinstance(other, Backpack):
            return NotImplemented
        return (
            self.name  == other.name and
            self.color == other.color and
            len(self.contents) == len(other.contents)
        )

    def __str__(self):
        """Return a nicely formatted description of this backpack."""
        return (
            f"Owner:\t{self.name}\n"
            f"Color:\t{self.color}\n"
            f"Size:\t{len(self.contents)}\n"
            f"Max Size:\t{self.max_size}\n"
            f"Contents:\t{self.contents}"
        )



# Problem 4: Write a 'ComplexNumber' class.
    class ComplexNumber:
    """A custom implementation of complex numbers a + bi."""

    def __init__(self, real, imag):
        """Initialize with real and imaginary parts."""
        self.real = real
        self.imag = imag

    def conjugate(self):
        """Return the complex conjugate: a − bi."""
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        """Return “(a+bj)” if imag ≥ 0, else “(a−bj)”, matching Python’s complex.__str__."""
        sign = "+" if self.imag >= 0 else "-"
        return f"({self.real}{sign}{abs(self.imag)}j)"

    def __abs__(self):
        """Return the magnitude √(a² + b²), so that abs(x) works."""
        return math.hypot(self.real, self.imag)

    def __eq__(self, other):
        """Two ComplexNumbers are equal iff real and imag parts match."""
        if not isinstance(other, ComplexNumber):
            return NotImplemented
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        """(a+bi) + (c+di) = (a+c) + (b+d)i."""
        return ComplexNumber(self.real + other.real,
                             self.imag + other.imag)

    def __sub__(self, other):
        """(a+bi) − (c+di) = (a−c) + (b−d)i."""
        return ComplexNumber(self.real - other.real,
                             self.imag - other.imag)

    def __mul__(self, other):
        """(a+bi)(c+di) = (ac−bd) + (ad+bc)i."""
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        return ComplexNumber(a*c - b*d,
                             a*d + b*c)

    def __truediv__(self, other):
        """(a+bi)/(c+di) = [(a+bi)(c−di)]/(c²+d²)."""
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        denom = c*c + d*d
        if denom == 0:
            raise ZeroDivisionError("division by zero complex number")
        real_part = (a*c + b*d) / denom
        imag_part = (b*c - a*d) / denom
        return ComplexNumber(real_part, imag_part)


    def test_ComplexNumber(a, b, tol=1e-9):
    """Compare ComplexNumber against built-in complex for various operations."""
    py = complex(a, b)
    my = ComplexNumber(a, b)

    # 1. Constructor
    if my.real != a or my.imag != b:
        print("__init__ failed: got", my.real, my.imag)

    # 2. Conjugate
    if my.conjugate().imag != py.conjugate().imag:
        print("conjugate() failed:", my.conjugate(), py.conjugate())

    # 3. __str__
    if str(my) != str(py):
        print("__str__ failed:", str(my), "!=", str(py))

    # 4. __abs__
    if abs(my) != abs(py):
        print("__abs__ failed:", abs(my), "!=", abs(py))

    # 5. __eq__
    if not (my == ComplexNumber(a, b)):
        print("__eq__ failed for identical values")
    if (my == ComplexNumber(a+1, b)):
        print("__eq__ false positive")

    # 6. Arithmetic
    ops = [
        ("add",     lambda x,y: x + y,        ComplexNumber.__add__),
        ("sub",     lambda x,y: x - y,        ComplexNumber.__sub__),
        ("mul",     lambda x,y: x * y,        ComplexNumber.__mul__),
        ("truediv", lambda x,y: x / y,        ComplexNumber.__truediv__),
    ]
    for name, pyop, myop in ops:
        py_res = pyop(py, complex(b, a))               # test with a second complex(b,a)
        my_res = myop(my, ComplexNumber(b, a))
        if not isinstance(my_res, ComplexNumber):
            print(f"{name} did not return ComplexNumber")
        if (abs(my_res.real - py_res.real) > tol or
            abs(my_res.imag - py_res.imag) > tol):
            print(f"{name} failed: {my_res} != {py_res}")

    print("Testing complete.")






if __name__ == "__main__":
    test_backpack()
    jp = Jetpack("Ava", "red")
    print(f"{jp.name!r} has a {jp.color} jetpack with capacity {jp.max_size} and fuel {jp.fuel}.")
    jp.put("map")
    jp.put("radio")
    jp.put("snack")     # should print "No Room!" because max_size is 2
    print("Contents after put:", jp.contents)

    jp.fly(3)
    print("Fuel after flying 3 units:", jp.fuel)
    jp.fly(8)           # should print "Not enough fuel!"
    print("Fuel after failed flight:", jp.fuel)

    jp.dump()
    print("Contents after dump():", jp.contents)
    print("Fuel after dump():", jp.fuel)
    
    test_ComplexNumber(3, -4)
    