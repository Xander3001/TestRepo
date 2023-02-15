def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


def fib(num):
    if num <= 1:  # For n equal to 0 or 1
        return num
    else:
        return fib(num-1) + fib(num-2) # For n greater than 1 i.e 2 or more
print(fib(10))