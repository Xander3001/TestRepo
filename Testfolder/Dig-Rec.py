# This function calculates the Fibonacci sequence number for a given integer positive number.

def fib(num):
    # If the given number is 0 or 1, return the number.
    if num <= 1:
        return num
    else:
        # Otherwise, return the sum of the two previous Fibonacci numbers.
        return fib(num-1) + fib(num-2)

# Print the 10th Fibonacci number.
print(fib(10))