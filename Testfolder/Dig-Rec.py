# This code defines a recursive function called "fib" that calculates the nth term of the Fibonacci sequence.
# The Fibonacci sequence is a series of numbers where each number is the sum of the preceding two numbers.
# The first two terms of the sequence are 0 and 1.
# The function takes a single parameter "num", which specifies the number of terms in the sequence to calculate.
# The function returns the nth term of the sequence.
# At the end of the file, the function is called with an argument of 10 and the result is printed to the console.

def fib(num):
    """
    Calculate the nth term of the Fibonacci sequence.

    Args:
        num (int): The number of terms in the sequence to calculate.

    Returns:
        int: The nth term of the Fibonacci sequence.

    """
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

print(fib(10))  # Should print 55 to the console.