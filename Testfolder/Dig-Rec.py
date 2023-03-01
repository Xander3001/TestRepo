def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


def fib(num):
    """
    Returns the nth number in the Fibonacci sequence using recursion.

    Parameters:
    num (int): The position of the desired Fibonacci number in the sequence.

    Returns:
    The nth Fibonacci number.

    """
    if num <= 1:  # base case
        return num
    else:
        # recursive case: adds up the previous two numbers in the sequence
        return fib(num-1) + fib(num-2)

# prints the 10th Fibonacci number
print(fib(10))