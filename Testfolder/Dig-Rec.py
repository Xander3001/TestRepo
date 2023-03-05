# This function implements the Fibonacci sequence recursively
# Takes a positive integer as input 
# Returns the value of the nth Fibonacci number

def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

# Prints the 10th Fibonacci number
print(fib(10))