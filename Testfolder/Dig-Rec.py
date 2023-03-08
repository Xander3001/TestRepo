'''
A function that returns the nth number in the Fibonacci sequence.
:param num: an integer representing the position of the desired number in the sequence
:return: an integer representing the value of the number in the position requested
'''

def fib(num):
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

print(fib(10))