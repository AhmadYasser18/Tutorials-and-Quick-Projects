# Data Structure
are data organization, management and storage format that enables efficient access and modification [Access, Insertion, Deletion, Search].

## Types:
- Primitve Data structures:
  - interger
  - float
  - character
  - pointers

- Non-Primitve Data structures:
  - Linear Data structures:
    - Array
    - Linked List
    - Stack
    - Queue
  - Non-linear Data structures:
    - Trees
    - Graphs

# Algorithms
While comparing/analyzing algorithms there are certain factors that determine how good an algorithm is.

**Efficiency:** could be represented by many metrics, including:
- Memory space
- Runtime
- Number of swaps/comparisons [for sorting algorithms]

**Runtime:** the simple definition is that refers to the amount of time it takes to solve a problem using an algorithm
  Primitive Operations: operations that take one time unit.
  As such the runtime could be defined as the number of primitive operations needed to solve the problem.

**Complexity:**
When searching for an element one of three cases could be faced: 
  - Best Case [Omega Notation]
  - Average/Middle Case [Theta Notation]
  - Worst Case [Big O Notation]
The main focus when comparing algorithms is the Big O notation.

### Big O notation:
O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)

Example (1):
```python

i = 0                               #1
while i < n:                        #n
    print(i)                        
    i+=1
            
for j in range(n):                  #n
    for k in range(n):              #n
         print(j+k)               __
            
                                    1 + n +n*n
```                                    
Complexity: O(n^2)

Example (2):
```python
i = 1                              #1
while i < n:                        
    print(i)
    i*=2                          #log n [base 2]
                                   _______
```                             
Complexity: O(log n)
            
Example (3):
```python
i = j = 0                         #1
while i < n:                        
    while j < n:
        print(i+j)
        j*=3                      #log n [base 3]
    i+=1                          #n    
                                   _______
                                   1 + n * log n
```
Complexity: O(n log n)
