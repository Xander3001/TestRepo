def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


# def fib(num):
#     if num <= 1:
#         return num
#     else:
#         return fib(num-1) + fib(num-2) #a function recursion
# print(fib(10))