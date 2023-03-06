def fib(num): # function to calculate fibonacci series
    '''
    This function generates the fibonacci series of a given number.
    '''
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10)) # prints the fibonacci series of the number 10.