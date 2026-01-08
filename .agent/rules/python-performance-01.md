---
trigger: always_on
---


**Hack 1: Leverage sets for membership testing**
------------------------------------------------

When you need to check whether an element exists within a collection, using a list can be inefficient – especially as the size of the list grows. Membership testing with a list (`x in some_list`) requires scanning each element one by one, resulting in linear time complexity (`O(n)`):

```
big_list = list(range(1000000))
big_set = set(big_list)
start = time.time()
print(999999 in big_list)
print(f"List lookup: {time.time() - start:.6f}s")

start = time.time()
print(999999 in big_set)
print(f"Set lookup: {time.time() - start:.6f}s")
```


**Time measured:**

*   List lookup: ~0.015000s
*   Set lookup: ~0.000020s

In contrast, sets in Python are implemented as hash tables, which allow for constant-time (`O(1)`) lookups on average. This means that checking whether a value exists in a set is significantly faster, especially when dealing with large datasets.

For tasks like filtering duplicates, validating input, or cross-referencing elements between collections, sets are far more efficient than lists. They not only speed up membership tests but also make operations like unions, intersections, and differences much faster and more concise.

By switching from lists to sets for membership checks – particularly in performance-critical code – you can achieve meaningful speed gains with minimal changes to your logic.

**Hack 2: Avoid unnecessary copies**
------------------------------------

Copying large objects like lists, dictionaries, or arrays can be costly in both time and memory. Each copy creates a new object in memory, which can lead to significant overhead, especially when working with large datasets or within tight loops.

Whenever possible, modify objects in place instead of creating duplicates. This reduces memory usage and improves performance by avoiding the overhead of allocating and populating new structures. Many built-in data structures in Python provide in-place methods (e.g. `sort`, `append`, `update`) that eliminate the need for copies.

```
numbers = list(range(1000000))
def modify_list(lst):
    lst[0] = 999
    return lst
start = time.time()
result = modify_list(numbers)
print(f"In-place: {time.time() - start:.4f}s")

def copy_list(lst):
    new_lst = lst.copy()
    new_lst[0] = 999
    return new_lst
start = time.time()
result = copy_list(numbers)
print(f"Copy: {time.time() - start:.4f}s")
```


**Time measured:**

*   In-place: ~0.0001s
*   Copy: ~0.0100s

In performance-critical code, being mindful of when and how objects are duplicated can make a noticeable difference. By working with references and in-place operations, you can write more efficient and memory-friendly code, particularly when handling large or complex data structures.

**Hack 3: Use `__slots__` for memory efficiency**
-------------------------------------------------

By default, Python classes store instance attributes in a dynamic dictionary (`__dict__`), which offers flexibility but comes with memory overhead and slightly slower attribute access.

Using `__slots__` allows you to explicitly declare a fixed set of attributes for a class. This eliminates the need for a `__dict__`, reducing memory usage – which is especially beneficial when creating many instances of a class. It also leads to marginally faster attribute access due to the simplified internal structure.

While `__slots__` does restrict dynamic attribute assignment, this trade-off is often worthwhile in memory-constrained environments or performance-sensitive applications. For lightweight classes or data containers, applying `__slots__` is a simple way to make your code more efficient.

```
class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
start = time.time()
points = [Point(i, i+1) for i in range(1000000)]
print(f"With slots: {time.time() - start:.4f}s")
```


**Time measured:**

*   With `__slots__`: ~0.1200s
*   Without `__slots__`: ~0.1500s

**Hack 4: Use `math` functions instead of operators**
-----------------------------------------------------

For numerical computations, Python’s `math` module provides functions that are implemented in C, offering better performance and precision than equivalent operations written in pure Python.

For example, using `math.sqrt()` is typically faster and more accurate than raising a number to the power of 0.5 using the exponentiation (`**`) operator. Similarly, functions like `math.sin()`, `math.exp()`, and `math.log()` are highly optimized for speed and reliability.

These performance benefits become especially noticeable in tight loops or large-scale calculations. By relying on the `math` module for heavy numerical work, you can achieve both faster execution and more consistent results – making it the preferred choice for scientific computing, simulations, or any math-heavy code.

![Use math functions instead of operators](https://blog.jetbrains.com/wp-content/uploads/2025/10/image-32.png)

[PyCharm](https://www.jetbrains.com/pycharm/) makes it even easier to take advantage of the `math` module by providing intelligent code completion. Simply typing `math.` triggers a dropdown list of all available mathematical functions and constants – such as `sqrt()`, `sin()`, `cos()`, `log()`, `pi`, and many more – along with inline documentation. 

This not only speeds up development by reducing the need to memorize function names, but also encourages the use of optimized, built-in implementations over custom or operator-based alternatives. By leveraging these hints, developers can quickly explore the full breadth of the module and write cleaner, faster numerical code with confidence.

```
import math
numbers = list(range(10000000))
start = time.time()
roots = [math.sqrt(n) for n in numbers]
print(f"Math sqrt: {time.time() - start:.4f}s")

start = time.time()
roots = [n ** 0.5 for n in numbers]
print(f"Operator: {time.time() - start:.4f}s")
```


**Time measured:**

*   `math.sqrt`: ~0.2000s
*   Operator: ~0.2500s

**Hack 5: Pre-allocate memory with known sizes**
------------------------------------------------

When building lists or arrays dynamically, Python resizes them in the background as they grow. While convenient, this resizing involves memory allocation and data copying, which adds overhead – especially in large or performance-critical loops.

If you know the final size of your data structure in advance, pre-allocating memory can significantly improve performance. By initializing a list or array with a fixed size, you avoid repeated resizing and allow Python (or libraries like NumPy) to manage memory more efficiently.

This technique is particularly valuable in numerical computations, simulations, and large-scale data processing, where even small optimizations can add up. Pre-allocation helps reduce fragmentation, improves cache locality, and ensures more predictable performance.

```
start = time.time()
result = [0] * 1000000
for i in range(1000000):
    result[i] = i
print(f"Pre-allocated: {time.time() - start:.4f}s")

start = time.time()
result = []
for i in range(1000000):
    result.append(i)
print(f"Dynamic: {time.time() - start:.4f}s")
```


**Time measured:**

*   Pre-allocated: ~0.0300s
*   Dynamic: ~0.0400s

**Hack 6: Avoid exception handling in hot loops**
-------------------------------------------------

While Python’s exception handling is powerful and clean for managing unexpected behavior, it’s not designed for high-frequency use inside performance-critical loops. Raising and catching exceptions involves stack unwinding and context switching, which are relatively expensive operations.

In hot loops – sections of code that run repeatedly or process large volumes of data – using exceptions for control flow can significantly degrade performance. Instead, use conditional checks (`if`, `in`, `is`, etc.) to prevent errors before they occur. This proactive approach is much faster and leads to more predictable execution.

Reserving exceptions for truly exceptional cases, rather than expected control flow, results in cleaner and faster code – especially in tight loops or real-time applications where performance matters.

```
numbers = list(range(10000000))
start = time.time()
total = 0
for i in numbers:
    if i % 2 != 0:
        total += i // 2
    else:
        total += i
print(f"Conditional: {time.time() - start:.4f}s")

start = time.time()
total = 0
for i in numbers:
    try:
        total += i / (i % 2)
    except ZeroDivisionError:
        total += i
print(f"Exception: {time.time() - start:.4f}s")
```


**Time measured:**

*   Conditional: ~0.3000s
*   Exception: ~0.6000s