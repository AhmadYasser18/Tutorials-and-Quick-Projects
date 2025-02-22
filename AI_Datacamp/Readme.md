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
3. Pass input_tensor to linear_layer to generate an output.

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

## Discovering activation functions
When using linear layers, each linear layer multiplies its input by its weights and adds biases. Two or more linear layers stacked in a row still effectively perform a linear operation.
However, linear layers are not the only layer type that can be added to a network. Using activation functions adds non-linearity to a network. This non-linearity grants networks the ability to learn more complex interactions between inputs X and targets y than only linear relationships. The output will no longer be a linear function of the input. The  the output of the last linear layer will be refered to as the "pre-activation" output, which'll pass to activation functions to obtain transformed output.

![image](https://github.com/user-attachments/assets/8d1e2abd-8991-4986-9c4f-13148d2d0f8c)

### The Sigmoid function
- Widely used for **binary classification** problems.
- Passing the input to a model with two linear layers returns a single output. This number is not yet interpretable as either category.
- The "pre-activation" output is passed the sigmoid and transform it to an output between zero and one.
- If the output is closer to one, we label it as class one. If it were less than 0.5, the prediction would be zero.

**Implementing sigmoid in PyTorch**
~~~python
import torch
import torch.nn as nn

input_tensor = torch.tensor([[6]])
sigmoid = nn.Sigmoid() #takes a one-dimensional input_tensor and returns an output bounded between zero and one.
output = sigmoid(input_tensor)
~~~
The sigmoid is commonly used as the **last step** in a neural network when performing binary classification. It is added as the last step in nn.Sequential(), and the last linear layer's output will automatically be passed to the sigmoid. *Note that a sigmoid as the last step in a network of only linear layers is equivalent to a logistic regression using traditional machine learning.*

~~~python'
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
~~~

### The Softmax function
Sigmoid is used for binary classification. For multiclass classification, involving more than two class labels, softmax is used. 
In this model softmax takes a n-dimensional pre-activation and generates an output of the same shape, one by n.

The output is a probability distribution because each element is between zero and one, and values sum to one.
In PyTorch, we use nn.Softmax(). dim equals -1 indicates that softmax is applied to input_tensor's last dimension. Similar to sigmoid, softmax can be the last layer in nn.Sequential.
~~~python'
import torch
import torch.nn as nn

input_tensor = torch.tensor([[x1, x2, x3]])

probabilities = nn.Softmax(dim= -1)
output = probabilities(input_tensor)
~~~

## Running a forward pass
When input data is passed through a neural network in the forward direction to generate outputs, or predictions, the input data flows through the model layers. At each layer, computations performed on the data generate intermediate representations, which are passed to each subsequent layer until the final output is generated. 
The purpose of the forward pass is to propagate input data through the network and produce predictions or outputs based on the model's learned parameters (weights and biases). This is used for both training and generating new predictions. 

In addition, there is there also a backward pass or backpropagation, which is the process by which layer weights and biases are updated during training. All this is part of something called a "training loop". This involves:
- Propagating data forward
- Comparing outputs to true values
- Propagating backwards to improve each layer's weights and biases.

It is repeated several times until the model is tuned with meaningful weights and biases. 

## Using loss functions to assess model predictions
In order to assess the differences between actual values and those predicted by the network **loss functions** are used.
The loss function tells us how good the model is at making predictions during training. It takes a model prediction, y-hat, and true label, or ground truth, y, as inputs, and outputs a float.

Loss is typically computed using a loss function. Consider loss function F, which takes ground truth y and prediction yhat as inputs and returns a numerical loss value. Possible values for a N-class case: true class labels y are integers (0 to N-1). yhat, on the other hand is the softmax function output. It is a tensor with the same dimensions as the number of classes N. 
### One-hot encoding concepts
We can use one-hot encoding to turn integer y to a tensor of zeros and ones, to compare like-for-like during evaluation. When y=0 and there are three classes, the encoded form of y=0 is [1,0,0]. The encoding is now also of shape 1 by 3, similar to yhat. We can represent this manual one-hot-encoding using a NumPy array.

**Transforming labels with one-hot encoding**
To prevent having to do this manually, we can aimport torch.nn.functional as F. \
~~~python
import torch.nn.functional as F
F.one_hot(torch.tensor(0), num_classes = 3)
#returns tensor([1,0,0])
F.one_hot(torch.tensor(1), num_classes = 3)
#returns tensor([0,1,0])
~~~
### Cross entropy loss in PyTorch

The loss function takes the scores tensor as input, which is the model prediction before the final softmax function, and the one-hot encoded ground truth label. It outputs a single float, the loss of that sample. The goal of training is to minimize the loss.
Passing the one-hot encoding along with the output predictions to a loss function. yhat is stored as the tensor "scores". 
**Cross-entropy loss*. This is the most used loss function for classification problems. PyTorch provides the handy CrossEntropyLoss() function shown. We start by 
- Define loss function  
- Pass it the **.double()** of the scores tensor and the one_hot_target tensor. This casts the tensors to a specific float data type that is accepted by the CrossEntropyLoss() function.
- The output shown is the loss value.
~~~python
from torch.nn import CrossEntropyLoss

scrores = tensor([[-0.1211, 0.1059]])
on_hot_target = tensor([[1, 0]])

criterion = CrossEntropyLoss() #Defining loss function
criterion(scores.double(), on_hot_target.double()
~~~

## Using derivatives to update model parameters
The goal of training is to minimize a loss function we define. Calculating derivatives is a core step for minimizing loss. Loss is high when a model is mispredicting while low when it's predicting correctly.

Think of the loss function as a valley. Each horizontal step (along x) involves gaining or losing some height (y). 
- **At steeper slopes**: a single step means losing or gaining a lot of elevation. Mathematically, the derivative (or slope) is high.
- **At gentler slopes**: a single step involves losing or gaining less elevation, meaning a smaller derivative.
- **Valley floor**: The bottom is flat, and elevation does not change at each step, the derivative is null. If the valley is our loss function, the function is at minimum when the derivative is null.

So how do gradients help minimize loss, tune layer weights and biases, and improve model predictions during training?

In deep learning, the term **"gradient"** is often used for derivatives. When a model is created, layer weights and biases are randomly initialized. Training looks as shown: 
- take a dataset with features X, and target y.
- run a forward pass using X and calculate loss by comparing model output, * yhat*, with y.
- compute gradients of the loss function and use them to update the model parameters with backpropagation so that weights are no longer random and biases are useful.
- repeat until the layers are tuned.

![image](https://github.com/user-attachments/assets/8005e370-412b-4cbc-8534-34482d84fd3d)

**Backpropagation**

Consider a network of three linear layers (L0,L1,L2) local gradients can be calculated with respect to each layer's parameters. 
- first calculate loss gradients with respect to L2,
- then L2 to L1
- repeat until the first layer is reached.

![image](https://github.com/user-attachments/assets/e8b05f1c-e0a2-4344-9a8c-9d75239212bd)

### **Implementing in PyTorch**

After running a forward pass on sample data, defining a loss function, using it to compare predictions with target values, **.backward()** is used to calculate gradients using this defined loss. This populates the .grad attributes of each layer's weights and biases. Each layer in the model can be indexed separately, beginning with a zero-index. Each layer has a weight, a bias, and the corresponding gradients.

~~~python
model = nn.Sequential(
    nn.Linear(16, 8),
    nn.Linear(8, 4),
    nn.Linear(4, 2),
    nn.Softmax(dim = -1)
)
prediction = model(sample)
#Calculating loss and computing gradient
criterion = CrossEntropyLoss() 
loss= criterion(prediction, target)

loss.backward()
#Accessing each layer's gradient
model[0].weight.grad, model[0].bias.grad
model[1].weight.grad, model[1].bias.grad
model[2].weight.grad, model[2].bias.grad 
~~~
 **Updating model parameters**

To update model parameters manually, access each layer gradient, multiply it by the learning rate, and then subtract this product from the weight. 

~~~python
lr = 0.001
#updating weights
weight = model[0].weight
weight_grad = model[0].weight.grad
weight -= lr*weight_grad

#updating biases
bias = model[0].bias
bias_grad = model[0].bias.grad
bias -= lr*bias_grad
~~~
**Convex and non-convex functions**

Some functions, such as the one on the left, have one minimum and [one only], called the **"global"** minimum. These functions are "convex". Some, "non-convex" functions have more than one "local" minimum. At a local minimum, the function value is lowest compared to nearby points, but points further away may be even lower. When minimizing loss functions, our goal is to find the global minimum of the non-convex function.

Convex:
![image](https://github.com/user-attachments/assets/ca49a7f1-1f43-43f0-8f88-7f0b463ed346)

non-convex:
![image](https://github.com/user-attachments/assets/a6f9197c-0f94-4698-a133-aca0a553b4af)

Loss functions used in deep learning are not convex. To find global minimum of non-convex functions **"gradient descent"** is used. PyTorch does this using "optimizers". The most common optimizer is [stochastic gradient descent] (SGD).
- SGD is instantiate using optim.
- .parameters() returns an iterable of all model parameters which are passed to the optimizer.
- a standard learning rate, "lr" is used here which is tunable.
- The optimizer calculates gradients, and updates model parameters automatically, by calling .step(). 
~~~python'
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()
~~~

