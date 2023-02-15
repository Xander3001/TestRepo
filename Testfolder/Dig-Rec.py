def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)
print(fib(10))


def fib(num):  #def:define (function) 
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2) #def of fib
print(fib(10))