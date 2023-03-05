# The code implementation below is the recursive implementation of the Fibonacci sequence. 

def fib(num):
    """
    A function that generates the nth number in the Fibonacci sequence recursively

    Parameters:
    num(int): A non-negative integer representing the index of the desired number in the sequence

    Returns:
    int: The nth number in the Fibonacci sequence.
 
    """
    if num <= 1:
        return num
    else:
        return fib(num-1) + fib(num-2)

print(fib(10))  # Expected output: 55

# The function accepts an integer argument representing the nth number in the Fibonacci sequence, 
# then returns that number using a recursive implementation. This specific implementation filters 
# numbers less than <= 1 and calculates the previous two Fibonacci numbers and sums them up to get the desired nth number.