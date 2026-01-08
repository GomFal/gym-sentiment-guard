---
trigger: always_on
---

**Hack 7: Use local functions for repeated logic**
--------------------------------------------------

When a specific piece of logic is used repeatedly within a function, defining it as a local (nested) function – also known as a closure – can improve performance and organization. Local functions benefit from faster name resolution because Python looks up variables more quickly in local scopes than in global ones.

In addition to the performance gain, local functions help encapsulate logic, making your code cleaner and more modular. They can also capture variables from the enclosing scope, allowing you to write more flexible and reusable inner logic without passing extra arguments.

This technique is particularly useful in functions that apply the same operation multiple times, such as loops, data transformations, or recursive processes. By keeping frequently used logic local, you reduce both runtime overhead and cognitive load.

```
def outer():
    def add_pair(a, b):
        return a + b
    result = 0
    for i in range(10000000):
        result = add_pair(result, i)
    return result
start = time.time()
result = outer()
print(f"Local function: {time.time() - start:.4f}s")

def add_pair(a, b):
    return a + b
start = time.time()
result = 0
for i in range(10000000):
    result = add_pair(result, i)
print(f"Global function: {time.time() - start:.4f}s")
```


**Time measured:**

*   Local function: ~0.4000s
*   Global function: ~0.4500s

**Hack 8: Use `itertools` for combinatorial operations**
--------------------------------------------------------

When dealing with permutations, combinations, Cartesian products, or other iterator-based tasks, Python’s `itertools` module provides a suite of highly efficient, C-optimized tools tailored for these use cases.

Functions like `product()`, `permutations()`, `combinations()`, and `combinations_with_replacement()` generate elements lazily, meaning they don’t store the entire result in your computer’s memory. This allows you to work with large or infinite sequences without the performance or memory penalties of manual implementations.

In addition to being fast, `itertools` functions are composable and memory-efficient, making them ideal for complex data manipulation, algorithm development, and problem-solving tasks like those found in simulations, search algorithms, or competitive programming. When performance and scalability matter, `itertools` is a go-to solution.

```
from itertools import product
items = [1, 2, 3] * 10
start = time.time()
result = list(product(items, repeat=2))
print(f"Itertools: {time.time() - start:.4f}s")

start = time.time()
result = []
for x in items:
    for y in items:
        result.append((x, y))
print(f"Loops: {time.time() - start:.4f}s")
```


**Time measured:**

*   `itertools`: ~0.0005s
*   Loops: ~0.0020s

**Hack 9: Use `bisect` for sorted list operations**
---------------------------------------------------

When working with sorted lists, using linear search or manual insertion logic can be inefficient – especially as the list grows. Python’s `bisect` module provides fast, efficient tools for maintaining sorted order using binary search.

With functions like `bisect_left()`, `bisect_right()`, and `insort()`, you can perform insertions and searches in `O(log n)` time, as opposed to the `O(n)` complexity of a simple scan. This is particularly useful in scenarios like maintaining leaderboards, event timelines, or implementing efficient range queries.

By using `bisect`, you avoid re-sorting after every change and gain a significant performance boost when working with dynamic, sorted data. It’s a lightweight and powerful tool that brings algorithmic efficiency to common list operations.

```
import bisect
numbers = sorted(list(range(0, 1000000, 2)))
start = time.time()
bisect.insort(numbers, 75432)
print(f"Bisect: {time.time() - start:.4f}s")

start = time.time()
for i, num in enumerate(numbers):
    if num > 75432:
        numbers.insert(i, 75432)
        break
print(f"Loop: {time.time() - start:.4f}s")
```


**Time measured:**

*   `bisect`: ~0.0001s
*   Loop: ~0.0100s

**Hack 10: Avoid repeated function calls in loops**
---------------------------------------------------

Calling the same function multiple times inside a loop – especially if the function is expensive or produces the same result each time – can lead to unnecessary overhead. Even relatively fast functions can accumulate significant cost when called repeatedly in large loops.

To optimize, compute the result once outside the loop and store it in a local variable. This reduces function call overhead and improves runtime efficiency, particularly in performance-critical sections of code.

This technique is simple but effective. It not only speeds up execution but also enhances code clarity by signaling that the value is constant within the loop’s context. Caching function results is one of the easiest ways to eliminate redundant computation and make your code more efficient.

```
def expensive_operation():
    time.sleep(0.001)
    return 42
start = time.time()
cached_value = expensive_operation()
result = 0
for i in range(1000):
    result += cached_value
print(f"Cached: {time.time() - start:.4f}s")

start = time.time()
result = 0
for i in range(1000):
    result += expensive_operation()
print(f"Repeated: {time.time() - start:.4f}s")
```


**Time measured:**

*   Cached: ~0.0010s
*   Repeated: ~1.0000s