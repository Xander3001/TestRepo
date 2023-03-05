def fib(num):
    """
    This function takes in the parameter 'num' which represents the index of the desired Fibonacci number to be returned. 
    The function uses recursion to compute the Fibonacci sequence by returning the sum of the two previous Fibonacci numbers.
    """
    if num <= 1: # Base case: if num is less than or equal to 1, return num as it is either the first or second Fibonacci number
        return num
    else: # Recursive case: return the sum of the two previous Fibonacci numbers
        return fib(num-1) + fib(num-2)

print(fib(10)) # Call the fib function and print the 10th Fibonacci number