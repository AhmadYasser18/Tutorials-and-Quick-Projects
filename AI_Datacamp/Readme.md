# Introduction to deep learning with PyTorch

Traditional machine learning relies on hand-crafted feature engineering. Deep learning models, on the other hand, using a "layered" network structure, 
are able to discover features from raw data, giving them that edge over traditional machine learning. This is known as **feature learning** or **representation learning**.
Deep learning is a subset of machine learning, where the fundamental model structure is a network of inputs, hidden layers, and outputs. A network can have one or many hidden layers. 
The original intuition behind deep learning was to create models inspired by how the human brain learns: through interconnected cells called neurons. This is why we continue to call deep learning models "neural" networks. These layered model structures require far more data to learn than other supervised learning models to derive patterns from the unstructured data in the manner we discussed. We are usually talking about at least hundreds of thousands of data points.
While there are several frameworks and packages out there for implementing deep learning algorithms, we'll focus on PyTorch, one of the most popular and well-maintained frameworks. In addition to being used by deep learning engineers in industry, PyTorch is a favored tool amongst researchers. Many deep learning papers are published using PyTorch. PyTorch is designed to be intuitive and user-friendly, sharing a lot of common ground with the Python library NumPy.

The PyTorch module can be imported by calling:
```python
import torch
```
The fundamental data structure in PyTorch is called a **tensor**. 
- similar to an array
- building block for neural networks.
- multidimensional

Tensors can be created from Python lists by using the 
```python
my_list = [1,2,3,4]
torch.tensor(my_list)
#Or using NumPy arrays
np_array = np.array(a)
np_tensor = torch.from_numpy(np_array)
```
 Some attribitues:
 ```python
tensor.shape  
tensor.dtype
tensor.device #displays which device the tensor is loaded on, such as a CPU or GPU
```
PyTorch tensors support several operations similar to NumPy arrays:
- add or subtract tensors, provided that their **shapes are compatible**. When shapes are incompatible, we get an error.
- Performing element-wise multiplication, multiplying each corresponding element from two arrays of the **same shape**,
- tensor transposition
- matrix multiplication
- tensor concatenation
- etc .....
  
Most operations available for NumPy arrays can be performed on PyTorch tensors.

# 

