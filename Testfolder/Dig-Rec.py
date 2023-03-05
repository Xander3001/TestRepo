# This function calculates the fibonacci sequence up to a given number "num"
# and returns the value at that number's index in the sequence.
def fib(num): 
    if num <= 1: # If input number is 0 or 1, return that number.
        return num
    else: # If input number is greater than 1, calculate the fibonacci sequence recursively
        return fib(num-1) + fib(num-2)
    
print(fib(10)) # Call the function and print the result