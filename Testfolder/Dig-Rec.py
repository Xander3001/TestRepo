# This is a recursive function that calculates the fibonacci sequence up to the given number.

def fib(num):
    # If the input is less than or equal to 1, return that number
    if num <= 1:
        return num
    else:
        # Otherwise, calculate the fibonacci sequence for num-1 and num-2 recursively
        return fib(num-1) + fib(num-2)

# Print the 10th fibonacci number
print(fib(10))