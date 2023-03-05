# Define a Fibonacci sequence function that takes in a parameter 'num'
def fib(num):
    # Check if num is less than or equal to 1
    if num <= 1:
        # If so, return the value of num
        return num
    else:
        # If num is greater than 1, return the sum of calling the fib function with num-1 and num-2
        return fib(num-1) + fib(num-2)
# Print the 10th number in the Fibonacci sequence
print(fib(10))