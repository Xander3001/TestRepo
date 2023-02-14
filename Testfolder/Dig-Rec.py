def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


# '''
# Recursive function that calculates the fibonaci series up to num: 14
# it calulates by calling itself when n=0 by n-1 then n-2 whenever a condition is met as it regresses upwards 
# '''
# 
# def fib(num):
#     if num <= 1:
#         return num
#     else:
#         return fib(num-1) + fib(num-2)
#         return(num)
# print(fib(14))