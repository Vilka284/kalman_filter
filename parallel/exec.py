from time import time


def func_time(function):
    def wrapped(*args):
        start_time = time()
        result = function(*args)
        end_time = time()
        total_time = end_time - start_time
        if total_time != 0:
            print(f"Execution of {function.__name__} took {round(total_time * 1000, 5)} ms")
        return result
    return wrapped
