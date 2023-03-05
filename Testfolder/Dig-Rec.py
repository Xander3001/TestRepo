# This function calculates the nth term of the Fibonacci sequence recursively
# Parameter: num - integer representing the nth term of Fibonacci sequence to calculate
# Returns: integer representing the value of the nth term of the Fibonacci sequence
def fib(num):
    # Base case when num is 0 or 1
    if num <= 1:
        return num
    else:
        # Recursive call to calculate the Fibonacci sequence
        return fib(num-1) + fib(num-2)

# Prints the 10th term of the Fibonacci sequence
print(fib(10))