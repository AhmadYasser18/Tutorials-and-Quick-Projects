# Introduction to deep learning with PyTorch
## Intro
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

## Creating neural networks

Neural networks are stacked inputs, hidden layers, and outputs.
While a network can have any number of hidden layers, we'll begin by building a basic, two-layer network with no hidden layers.

![image](https://github.com/user-attachments/assets/40b03d95-b009-4486-921e-305950182b2a)

### Using torch.nn()
1. The first layer of this network will be an input layer. Assuming an input_tensor of shape 1 by 3. Think of it as one row with three "features" or "neurons".
2. Pass the input_tensor to a special kind of layer called a **linear layer**.  A linear layer takes an input tensor, applies a linear function to it, and returns an output. nn.linear takes two arguments: 
  -  **in_features**: the number of features in our input (three),
  -  **out_features**: specifying the desired size of the output tensor (in this case two).
4. Pass input_tensor to linear_layer to generate an output.

~~~python
import torch.nn as nn

input_tensor = torch.tensor([[0.3471, 0.4547, 0.2356]])
#Defining linear layer
linear_layer = nn.Linear(in_features = 3, out_features = 2)
#Pass through linear layer
output = linear_layer(input_tensor)
~~~
Notice that the output will have two features or neurons, due to the out_features specified in our linear_layer.

Consider the linear_layer object we created. Each linear layer has a set of *weights* and *biases* associated with it.
**What operation does nn.Linear() perform?**
When input_tensor is passed to linear_layer, the linear operation performed is a matrix multiplication of input_tensor and the weights, followed by adding in the bias.
![image](https://github.com/user-attachments/assets/4a4fbaf4-b8b1-472c-a6fa-d22fdafd10c9)

When nn.Linear() is called , weights and biases are initialized randomly, so they are not yet useful. Later on, these weights and biases can be tuned so that the linear operation output is meaningful.

And just like that, a two-layer network is built. It took a 1 by 3 input as the first layer, a linear layer with specific arguments as the second layer, and returned a 1 by 2 output. *Note that networks with only linear layers are called "fully connected networks". Linear layers have connections (or arrows) between each input and output neuron, making them fully connected.*

### Stacking layers with nn.Sequential()
In case of stacking multiple layers, **nn.Sequential()** is used. The code below shows three linear layers stacked within nn.Sequential(). This model takes input, passes it to each linear layer in sequence, and returns output. The first layer takes input with 10 features and outputs a tensor with 18 features. The second layer takes an input of size 18 (the output size of the first layer) and outputs a tensor of size 20. The final layer takes input with the second layer's output size 20 and outputs a tensor of size 5.
~~~python'
model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20)
    nn.Linear(20, 5)
)
#Passing input_tensor to model
output_tensor = model(input_tensor)
~~~
Having one row of data called input_tensor, with ten "features", or neurons. Recall we set the first argument of our first linear layer to ten, to be compatible with this shape. We now pass input_tensor to our multi-layer model to obtain output, just as done before using a single linear layer. Again, output is not meaningful until each layer has tuned weights and biases.

