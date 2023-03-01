def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


"""
The function `fib(num)` takes an integer `num` as input and returns the nth Fibonacci number.
If `num` is less than or equal to one, the function returns `num`.
Otherwise, it recursively calculates the previous two Fibonacci numbers and returns their sum.
The program then calls the function with an argument of 10 and prints the result, which is the 10th Fibonacci number.
"""