# This code calculates the 10th Fibonacci number using recursion


def fib(num):
    """
    Calculates the nth Fibonacci number using recursion
    :param num: the index of the desired Fibonacci number
    :return: the value of the Fibonacci number at that index
    """
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)


print(fib(10)) # Prints the 10th Fibonacci number


# Output: 55