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
**What operation does `nn.Linear()` perform?**
When input_tensor is passed to linear_layer, the linear operation performed is a matrix multiplication of input_tensor and the weights, followed by adding in the bias.

![image](https://github.com/user-attachments/assets/4a4fbaf4-b8b1-472c-a6fa-d22fdafd10c9)

When `nn.Linear()` is called , weights and biases are initialized randomly, so they are not yet useful. Later on, these weights and biases can be tuned so that the linear operation output is meaningful.

And just like that, a two-layer network is built. It took a 1 by 3 input as the first layer, a linear layer with specific arguments as the second layer, and returned a 1 by 2 output. *Note that networks with only linear layers are called "fully connected networks". Linear layers have connections (or arrows) between each input and output neuron, making them fully connected.*

### Stacking layers with nn.Sequential()  
In case of stacking multiple layers, `nn.Sequential()` is used. The code below shows three linear layers stacked within `nn.Sequential()`. This model takes input, passes it to each linear layer in sequence, and returns output. The first layer takes input with 10 features and outputs a tensor with 18 features. The second layer takes an input of size 18 (the output size of the first layer) and outputs a tensor of size 20. The final layer takes input with the second layer's output size 20 and outputs a tensor of size 5.
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
The sigmoid is commonly used as the **last step** in a neural network when performing binary classification. It is added as the last step in `nn.Sequential()`, and the last linear layer's output will automatically be passed to the sigmoid. *Note that a sigmoid as the last step in a network of only linear layers is equivalent to a logistic regression using traditional machine learning.*

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
In PyTorch, we use `nn.Softmax()`. dim equals -1 indicates that softmax is applied to input_tensor's last dimension. Similar to sigmoid, softmax can be the last layer in nn.Sequential.
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

After running a forward pass on sample data, defining a loss function, using it to compare predictions with target values, `.backward()` is used to calculate gradients using this defined loss. This populates the .grad attributes of each layer's weights and biases. Each layer in the model can be indexed separately, beginning with a zero-index. Each layer has a weight, a bias, and the corresponding gradients.

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

Some functions, such as the one on the left, have one minimum and *one only*, called the **"global"** minimum. These functions are "convex". Some, "non-convex" functions have more than one "local" minimum. At a local minimum, the function value is lowest compared to nearby points, but points further away may be even lower. When minimizing loss functions, our goal is to find the global minimum of the non-convex function.

Convex:
![image](https://github.com/user-attachments/assets/ca49a7f1-1f43-43f0-8f88-7f0b463ed346)

non-convex:
![image](https://github.com/user-attachments/assets/a6f9197c-0f94-4698-a133-aca0a553b4af)

Loss functions used in deep learning are not convex. To find global minimum of non-convex functions **"gradient descent"** is used. PyTorch does this using "optimizers". The most common optimizer is [stochastic gradient descent] (SGD).
- SGD is instantiate using optim.
- .parameters() returns an iterable of all model parameters which are passed to the optimizer.
- a standard learning rate, "lr" is used here which is tunable.
- The optimizer calculates gradients, and updates model parameters automatically, by calling .step(). 
~~~python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()
~~~

## Writing Training loop

Training a neural network
- create a model
- choose a loss function
- create a dataset
- define an optimizer
- run a training loop over each element of the dataset
  - calculate the loss
  - compute local gradients
  - update model parameters
  
This final step is repeated a certain number of times, and is called the training loop. In scikit-learn, the training loop is located in the **.fit()** method of the model. We'll have to implement our own training loop in PyTorch.

Assuming a regeression oriblem:
Since the target is not a category but a continuous quantity, mostly the same neural network structure for regression and classification problems are used but:
- softmax or sigmoid activation can't be used for the last activation function as they are only used in classification problems.
- the last layer will be a linear layer.
- a new loss function is required, because cross entropy loss is also solely used for classification problems.

### Mean Squared Error Loss
Mean squared error (MSE) loss can be used for regression problems. *It is used in scikit-learn when calling the .fit method on a linear regression model.* The MSE loss is the mean of the squared difference between predictions and ground truth, as shown in this Python implementation. 
~~~python
def mse(pred, target):
    return np.mean((pred-target)**2)
~~~

The **nn.MSELoss** function is also available in PyTorch and can be used as a criterion similarly to cross entropy loss, where prediction and targets must be a float tensor.
~~~python
criterion = nn.MSELoss()
# Prediction and target are float numbers
loss = criterion(prediction, target)
~~~

Putting everything together:
- Having two NumPy arrays, "features" and "target", containing  data and labels.
- Pass these to a class called **"TensorDataset"** which is used to organize the features and targets into the right data types(floats). This casting is required, and float is the data type used by the parameters of our model.
- The dataset is loaded into a different class: **DataLoader()**, used to create "batches" of data, that are passed through the model in each forward/backward pass. Selection of batch size is customizable depending on the use case.
- Create the model next. For a dataset having four input features and one target. Because this is a regression problem, one-hot encoding isn't needed and the final linear layer outputs a single float.
- Finally, create the mean square loss criterion and the optimizer. A ten to minus three learning rate is a good default learning rate for most deep learning problems.
- **Looping:**
   -  For each epoch, we loop through the dataloader.
   -  Each iteration of the dataloader provides a batch of samples.
   -  Before the forward pass, gradients are set to zero using optimizer.zero_grad(), because the optimizer stores gradients from previous steps by default.
   -  Features and targets are obtained from each sample of the dataloader.
   -  Features are used for the forward pass of the model, and the target is used for the loss calculation.
   -  The optimizer is used to update the parameters of the model.
  
~~~python
#Create dataset and dataloader
dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(target).float())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#Create model
model = nn.Sequential(nn.Linear(4,2),
                      nn.Linear(2,1)
)
#Create the loss and optimizer 
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#Loop through the dataset

for epoch in range(num_epochs):
    for data in dataloader:
        #Set the gradients to zero
        optimizer.zero_grad()
        #Get feature and target from dataloader
        feature, target = data
        #Run a forward pass
        pred = model(features)
        #Compute loss and gradients
        loss = criterion(pred, target)
        loss.backward()
        #Update parameters
        optimizer.step() 
~~~
Looping through the entire dataset once is called an **epoch** and we train over multiple epochs, indicated by num_epochs.

## Activation functions between layers  
### Limitations of the sigmoid and softmax function
The outputs of the sigmoid are bounded between zero and one, meaning that for any input, the output will be greater than zero and less than one. Unlike the softmax function, it can be used anywhere in the network. 

![image](https://github.com/user-attachments/assets/28dfca75-6ad5-49be-bf34-6e2c4ceec994)

When plotting out the derivatives of the sigmoid function, notice that the gradients are always low and approach zero for low and high values of x. This behavior is called **saturation**. This property of the sigmoid function creates a challenge during backpropagation because each local gradient is a function of the previous gradient. For high and low values of x, the gradient will be so small that it can prevent the weight from changing or updating. This phenomenon is called vanishing gradients and makes training the network challenging. 
Because each element of the output vector of a softmax activation function is also bounded between zero and one, the softmax also saturates.

### ReLU Function  
The rectified linear unit or ReLU function outputs the maximum between its input and zero, as shown by the graph. For positive inputs, the output of the function is equal to the input. For strictly negative outputs, the output of the function is equal to zero. This function does not have an upper bound and the gradients do not converge to zero for high values of x, which overcomes the vanishing gradients problem. In PyTorch, the ReLU function may be called using
~~~python
relu = nn.ReLU()
~~~
ReLU is a good default choice of activation for many deep learning problems.

![image](https://github.com/user-attachments/assets/39cab2c1-e47a-4c89-ada2-52157210c990)

**Leaky ReLU**  
The leaky ReLU is a variation of the ReLU function. For positive inputs, it behaves similarly to the ReLU function. For negative inputs, however, it multiplies them by a small coefficient (defaulted to zero.zero-one in PyTorch). By doing this, the leaky ReLU function has non-null gradients for negative inputs. In PyTorch, the leaky ReLU function is called using:
~~~python
leaky_relu = nn.LeakyReLU(neagtive_slope=0.05)
~~~
The negative_slope parameter indicates the coefficient by which negative inputs are multiplied.

## 1. A deeper dive into neural network architecture
When each neuron of the layer is connected to each neuron of the previous layer, they are called fully connected layers. Each neuron of a linear layer will compute a linear operation using all the neurons of the previous layer. Therefore, a single neuron has N+1 learnable parameters, with N being the output dimension of the previous layer, plus one for the bias.

Fully connected neural networks are made of three types of layers: 
- an input layer: which contains the dataset features
- an output layer: which contains the model predictions
- hidden layers: which are any layers that are not the input or the output.

When designing a neural network, the dimensions of the input layer and output layer are fixed for us. The number of neurons in the input layer is the number of features in our dataset, and the number of neurons in the output layer is the number of classes we want to predict. However, we can add as many hidden layers as we want, as long as the input dimension of a layer matches the output dimension of the previous one. Increasing the number of hidden layers increases the number of parameters in the model which is also called increasing model capacity. Models with more capacity can work with more complex datasets.

A good metric to assess the capacity of our model is the total number of parameters. Let's consider this model made of two layers. 
~~~python
model = nn.Sequential(nn.Linear(8,4),
                      nn.Linear(4,2)
)
~~~
We can manually count the parameters: the first layer has 4 neurons, and each of them has 9 learnable parameters so it contains 36 parameters. The second layer has 2 neurons, each of them having 5 learnable parameters. In total, this model has 46 learnable parameters.
The same calculation can be done in PyTorch using the **.numel** method of tensors, which returns the number of elements in the tensor. By looping over the model parameters, we can add the number of elements per parameter to the total variable, and verify that it also adds up to 46.
~~~python
total = 0
for parameter in model.parameters():
    total+= parameter.numel()
~~~
When iterating on a deep learning model, we will often play with the number of hidden layers. However, remember that a bigger model will take longer to train.

## Learning rate and momentum  
The model's architecture impacts the training process and the model's performance. Training a neural network involves solving an optimization problem where the goal is to minimize the loss function by tweaking the model's parameters. 
To do this, an algorithm called [stochastic gradient descent], or SGD was used. The optimizer takes:
- the model's parameters
- the learning rate: which controls the step size taken by the optimizer
- the momentum: which controls the inertia of the optimizer.

Finding good values for the learning rate and momentum is critical, as bad values can lead to longer training times, or bad overall performance. 

**Impact of the learning rate:**  
- optimal learning rate  
Taking the below function as an example,the SGD optimizer runs for ten steps and starts at x = -2. After ten steps:
 - the optimizer has almost found the minimum of the function.
 - the step size taken by the optimizer is getting smaller as it is getting closer to x=0. This is because the step size equals the gradient multiplied by the learning rate. At around zero, this function is not as steep and therefore the gradient is smaller.
![optimum](https://github.com/user-attachments/assets/9d5ccc0d-0b50-446f-9a1f-41b08ee395ff)

- small learning rate  
![small](https://github.com/user-attachments/assets/eedb74ee-2a72-4928-9ef5-e63955aa312b)

For a learning rate ten times smaller, the minimum of the function is still far away after ten steps. The optimizer will take much longer to find the function's minimum.

- high learning rate
 ![high](https://github.com/user-attachments/assets/bd80f8cf-7067-4ec9-91ab-2e7573345da4)

For a high value for the learning rate, the optimizer cannot find the minimum and bounces back and forth on both sides of the function.

Typical learning rate values range from ten raised to -2, to ten raised to -4.


**Impact of momentum**  

As loss functions are non-convex. One of the challenges when trying to find the minimum of a non-convex function is getting stuck in a local minimum. 
For a null momentum on, the optimizer gets stuck in this first dip of the function, which is not its global minimum.
![null](https://github.com/user-attachments/assets/5145085f-7cd0-47f9-a43c-4d3210168310)
However, when using a momentum of 0.9, the minimum of the function can be found. This parameter provides momentum to the optimizer enabling it to overcome local dips. The momentum keeps the step size large when previous steps were also large, even if the current gradient is small.
![non null](https://github.com/user-attachments/assets/cfda3864-5a38-45d8-a356-122e821077ce)

Momentum usually ranges from 0.85 to 0.99.

## Layers initialization, transfer learning and fine tuning
**Layer initialization**  

Similarly to data normalization, the weights of a linear layer are also normalized. For example creating a small linear layer and looking at the minimum and maximum values of its weights, it's observed that the layer weights are initialized to small values.

In general, when training a neural network, weights of each layer are initialized to small values. The output of a neuron in a linear layer is a weighted sum of the outputs of the previous layer. Keeping both the input data and the layer's weights small ensures that the outputs of layers remain small. A layer can be initialized in different ways, for example using the uniform distribution. 
~~~python
import torch.nn as nn

layer = nn.Linear(64, 128)
print(layer.weight.min(), layer.weight.max()) 
~~~

Pytorch provides an easy way to initialize layers weights with the `nn.init` module.
~~~python
import torch.nn as nn

layer = nn.Linear(64, 128)
nn.init.uniform_(layer.weight)
~~~

### Transfer learning and fine tuning

In practice, machine learning engineers are rarely training a model from randomly initialized weights. Instead they rely on a very powerful concept called **transfer learning**. Transfer learning consists in taking a model that was trained on a first task and reuse for a second task. 

Instead of training a model using randomly initialized weights, weights can be loaded from the first model and used as a starting point to train on a new dataset. Saving and loading weights in pytorch can be done using the **torch.save** and the **torch.load** These functions works on any type of pytorch objects, whether it's a single layer or a full model.

Sometimes, the second task is similar to the first task and it is required to perform a specific type of transfer learning, called **fine-tuning**. In this case, weights are loaded from a previously trained model, but the model is trained with a smaller learning rate. A part of a network can be trained, if it's decided some of the network layers do not need to be trained and choose to freeze them. A rule of thumb is to freeze early layers of the network and fine-tune layers closer to the output layer. This can be achieved by setting each parameter's requires_grad attribute to False. Here, we use the model's named_parameters() method, which returns the name and the parameter itself. We set requires_grad of the first layer's weight to False.
~~~python
import torch.nn as nn

model = nn.Sequential(Linear(64, 128),
                      Linear(128, 256))

for name, param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad = False
~~~

## Loading data  
Having data stored in a csv file, it's loaded using pandas `read_csv()`.
We can use column indexing to return required columns. These will be the features, and can be casted to a NumPy array.
~~~python
features = dataset.iloc[:,x1:xn] #from column 1 to column n-1
X = features.to_numpy()
~~~
Similarly, the target column is extracted and stored as an array.
~~~python
target = dataset.iloc[:,xn] 
y = target.to_numpy()
~~~

`torch.utils.data` provides the **`DataLoader()`** and **`TensorDataset()`** classes, which are used to set data up for training a PyTorch model. TensorDataset() is a class that can be used to create a PyTorch dataset for tensors. Passing features X and target y to `TensorDataset()` creates a PyTorch dataset class conveniently. 
~~~python
import torch
from torch.utils.data import TensorDataset

#Instantiate dataset class
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

#Accessing a sample
sample = dataset[0] #the result is a tuple
input_sample, label_sample = sample
~~~

Once the dataset object is created using `TensorDataset()`, it's passed to the `DataLoader()`. Defining some customizable parameters:
- **batch_size**: determines how many samples are taken from the dataset per iteration.
- **shuffle**: tells the dataloader to shuffle the data on each iteration.

~~~python
import torch
from torch.utils.data import DataLoader

#Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle = True)
~~~
Each element of dataloader is a tuple, which can be unpacked batch_inputs and batch_labels. 

## Evaluating model performance  
When starting a machine learning project, datasetis is to be split into three subsets: [training], [validation], and [testing]. 
- During training, model parameters are adjusted (weights and biases).
- During validation, hyperparameters are tuned, such as learning rate and momentum.
- The test dataset is used only once to calculate final metrics.

### Model evaluation metrics  
For now, the focus will be on evaluating the following metrics in PyTorch: 
- loss during training and validation
- accuracy during training and validation.

In classification tasks, accuracy is a measure of how well a model correctly predicts ground truth labels.

**Calculating training loss**  
Loss is calculated by summing up loss at each iteration of the dataloader. At the end of each epoch, the mean value of the training loss is calculated. In PyTorch: 
- begin by initializing training loss to zero.
- iterate through the training dataloader, run a forward pass
- compute the loss.
- calculate gradients and update weights
- add the current loss to the previous training losses.

The loss tensor's .item() method returns the Python number contained in the tensor. One complete loop through the data loader is one epoch. Mean loss is calculated by dividing total training_loss by the dataloader's length: the number of batches in the dataset.
~~~python
training_loss = 0.0
for i, data in enumerate(trainloader,0):
    #Run forward pass
    .......
    #Calculate loss
    loss = criterion(outputs, labels)
    #Calculate gradients
    .......
    #Calculate and sum the loss
    training_loss+=loss.item()

epoch_loss = training_loss/len(trainloader)
~~~

**Calculating validation loss**  
A similar approach is taken to calculate validation loss. After each training epoch, we then iterate over the dataloader containing the validation dataset. 
- The model.eval is used to put the model in evaluation mode, because some layers in PyTorch models behave differently at training versus validation stages.
- add a Python context with torch.no_grad, indicating we will not be performing gradient calculation in this epoch.

Validation loss is then calculated similarly to training loss. We set the model back to training mode at the end of the validation epoch, so we can run another training epoch.
~~~python
val_loss = 0.0
model.eval() #putting model in evaluatoin mode
with torch.no_grad(): #Speedpup forward pass
 for i, data in enumerate(trainloader,0):
     #Run forward pass
     .......
     #Calculate loss
     loss = criterion(outputs, labels)
     val_loss+=loss.item()
 
epoch_loss = val_loss/len(trainloader)
model.train()
~~~
**Overfitting**  
Keeping track of validation and training losses during training helps in detecting overfitting. Overfitting occurs when the model stops generalizing and performance on the validation dataset degrades. This figure shows overfitting happens when validation loss is high but training loss is not.

![image](https://github.com/user-attachments/assets/a9f23411-4657-4b52-b9bc-3a767830aa31)

**Calculating accuracy with torchmetrics**  
In addition to loss, keep track of other metrics to evaluate how well the model is at predicting correct answers. To do so, **torchmetrics** is used. 

For classification, it's used to create an accuracy metric. On each iteration of dataloader, we call this metric using model outputs and ground truth labels. The accuracy metric takes probabilities and single number labels as inputs. The output variable here would be the probabilities returned by the softmax function. If the labels contain one-hot encoded classes, we'll need the argmax function to obtain numbers instead of one-hot vectors. At the end of the epoch, we calculate total accuracy using the metric's .compute() method. Finally, we use .reset() to reset the metric for the next epoch. Accuracy is calculated in the same way for training and validation.

~~~python
import torchmetrics

metric = torchmetrics.Accuracy(task="multiclass", num_classes=3) 

for i, data in enumerate(trainloader,0):
    features, labels =data
    outputs = model(features)
    #Calculate accuracy over batch
    acc = metric(outputs, labels.argmax(dim=-1)
#Calculate accuracy over whole epoch
acc.compute()
~~~

## Overfitting
**Reasons for overfitting**  
Overfitting happens when the model does not generalize to unseen data. If model is not trained correctly, it will start to memorize the training data, which leads to good performance on the training set but poor performance on the validation set. Several factors can lead to overfitting: 
- a small dataset
- a model with too much capacity
- large values of weights

Countering overfitting:
- model size can be reduced
- adding a new type of layer **dropout**
- using weight decay to force the parameters to remain small.
- Getting more data or using data augmentation

**"Regularization" using a dropout layer**  
A common way to fight overfitting is to add dropout layers to our neural network. Dropout is a "regularization" technique where randomly, a fraction of input neurons is set to zero at each update, effectively "dropping" them out. Corresponding connections are temporarily removed from the network, making the network less likely to overly rely on specific features. 
~~~python
import torch.nn as nn

model = nn.Sequential(Linear(8, 4),
                      nn.ReLU(),
                      nn.Dropout(p=0.5)) #p indicates the probability of setting a neuron to zero.
features = torch.randn((1, 8))
model(i)
~~~
Dropout can be added to models as shown. The  Usually, dropout layers are added after activation functions. Dropout layers behave differently between training and evaluation thus switching the model mode must not be forgotten. model.eval() / model.train()

**Regularization with weight decay**
~~~python
optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay= 1e-4)
~~~
In PyTorch, weight decay can be added to the optimizer as shown. It is controlled by the **weight_decay** parameter (ranging between 0 and 1 but is typically very small). When the optimizer's weight_decay parameter is set, it adds an additional term to the parameter update step that encourages smaller weights. This regularization term is proportional to the current value of the weight, and it is subtracted from the gradient during backpropagation. The weight decay term effectively penalizes large weights and helps prevent overfitting. The higher we set this parameter, the less likely our model is to overfit, so the model can generalize better to new data.

**Data augmentation**  
Getting more data can be costly. However, researchers have found a way to artificially increase the size and diversity of their dataset by using data augmentation. Data augmentation is commonly applied to image data, which can be rotated and scaled, so that different views of the same face become available as "new" data points.

## Improving model performance

**- Steps to maximize performance**  
When facing a new deep learning challenge, a series of steps should be followed to make sure that the best possible outcomewe is reached.
1. create a model that can overfit the training set. This ensures that the problem is solvable as well as setting a performance baseline to aim for with the validation set.
2. reduce overfitting to increase performance on the validation set.
3. slightly adjust the different hyperparameters to make sure the best possible performance is achieved.

**Step 1: overfit the training set**  
Before overfitting the whole training set, it is recommended to overfit a single data point. This can be done by slightly modifying the training loop such that it does not iterate through the dataloader anymore. 
~~~python
features, labels = next(iter(trainloader))
for i range(1e3):
  outputs = model(features)
  loss = criterion(outputs, labels)
  optimizer.step()
~~~
Overfitting a single data point should give an accuracy of one and a loss close to zero. It should also help find any potential bugs in the code if these values are not reached.  At this stage, it's best not to experiment with the model architecture but rather use an existing one if possible. The model should be large enough such that it can overfit the training set. Similarly, the different hyperparameters such as the learning rate should be kept to their defaults.

**Step 2: reduce overfitting**  
After ensuring that the problem is solvable with deep learning.,it's required to create a model that generalizes well to maximize the validation accuracy. Experiment with:
- dropout
- data augmentation
- weight decay
- or reducing the model capacity.

Keep track of the different parameters and the corresponding validation accuracy for each set of experiments. Commonly, the different overfitting reduction strategies lead to a significant drop in performance, as shown here. On the left is the original model overfitting the training set. On the right is the results of a model trained to reduce overfitting. The performance on both validation and training sets are much worse. This highlights the importance of keeping track of the different metrics when trying to find the best model.
![image](https://github.com/user-attachments/assets/9d2ee4c3-e578-48b0-bfea-775b9937ffdd)
![image](https://github.com/user-attachments/assets/f862a493-8149-4d4e-b3bf-7b3430d4def9)

**Step 3: fine-tune hyperparameters**  
Once an accepted performance is reached, fine-tuning can be done to the different hyperparameters. If sufficient computational capacity is available, a grid search over the different hyperparameters can be done. Usually, this is done on the optimizer parameters such as the learning rate or momentum. The grid search will use values of the parameters at a constant interval. For example, experiment with every momentum value between 0.85 and 0.99 with a constant interval. The learning rate is chosen between 10e-2 and 10e-6. 
~~~python
for factor range(2,6):
  lr = 10** - factor
~~~
Random search can also be used, which randomly samples parameters between intervals. 
~~~python
factor = np.random.uniform(2,6)
lr = 10** - factor
~~~
The random search approach often leads to better results.
