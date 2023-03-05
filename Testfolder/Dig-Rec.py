# This code calculates the 10th number in the Fibonacci sequence using recursion.

def fib(num):  # This is a function that takes a single argument, num.
    if num <= 1:  # If num is less than or equal to 1, return num.
        return num
    else:  # Otherwise, return the sum of the previous two numbers in the sequence.
        return fib(num-1) + fib(num-2)

print(fib(10))  # Call the function with num = 10 and print the result.