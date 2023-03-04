# The code is an implementation of the Fibonacci sequence
# It takes a number as input and returns the corresponding Fibonacci number

def fib(num):
    """
    Compute the Fibonacci of given number
    
    :param num: int, input number to compute Fibonacci of
    :return: int, Fibonacci number of input 
    """
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

print(fib(10)) # prints the 10th Fibonacci number