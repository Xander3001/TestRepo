'''
Function: fib

Arguments: num: integer value representing the position of the fibonacci sequence to be returned

Returns: integer value representing the fibonacci number at the given position.
'''

def fib(num):
    '''
    This function returns the fibonacci number at the given position.
    '''
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

print(fib(10))