def fib(num):       # function to calculate fibonacci number
    """
    Function that calculates the nth Fibonacci number recursively.
    
    Args:
    num (int): integer value for nth Fibonacci number
    
    Returns:
    int: value of the nth Fibonacci number
    
    """
    if num <= 1:    # base case for recursion
        return num
    else:
        return fib(num-1) + fib(num-2)   # recursive function call to calculate Fibonacci number
print(fib(10))      # printing the 10th fibonacci number