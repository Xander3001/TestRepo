def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


# The following function calculates the nth Fibonacci number using recursion.
def fib(num):
    """
    This function receives an integer, 'num', and returns the nth Fibonacci number.
    
    If the received number is less than or equal to 1, the function simply returns that number.
    Otherwise, the function recursively calls itself, calculating the sum of the two previous Fibonacci numbers.
    
    Args:
    - num: An integer representing the desired Fibonacci number to be calculated
    
    Return:
    - An integer representing the requested Fibonacci number.
    """
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

# This line calls the fib function, passing 10 as its argument, and prints the returned value.
print(fib(10))