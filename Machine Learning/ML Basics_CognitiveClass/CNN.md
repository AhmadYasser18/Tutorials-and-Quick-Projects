# Convolutional Neural Networks
**Image classification** is the process of taking an image and getting a computer to automatically sort it into a "class".
Computers interpret a digital image as a matrix of numbers representing the intensity values at every pixel. Intensity values are values that a single pixel can take on, ranging from 0 to 255. These valuesare used to classify the image. Let X represent the image to be classified, with RGB images, X will be a 3D array of intensity values. For grayscale images X is a 1D array.

A Convolutional Neural Network (CNN) is a function that predicts the class of an image, similar to how a function takes in a number as an input and produces an output. 
A CNN uses the entire image array as an input and outputs the class.

**Linear Classifier**  
A simple classifier is a linear classifier that uses a line, specifically a plane z to classify the data. Let's represent each image as a two-dimensional point. We also overlay the plane at z=0 with the following line:
![image](https://github.com/user-attachments/assets/e341fcae-c420-4861-8c85-2c34c7f90339)

A plane can't always separate data. In the below figure X_7, is misclassified. In this case, the dataset is not linearly separable.
![image](https://github.com/user-attachments/assets/45666529-c20c-45ca-94c4-6a34b2bd3678)

**Training Linear Classifier**  
Training is where the best linear classifier is obtained. More precisely, the learnable parameters w_1, w_2 and b. This is done by minimizing a cost function, a function that outputs the number of incorrectly classified samples as a functino of the learnable parameters: Cost(w_1, w_2, b).
Consider the following cost functions: Cost(w_1=1,w_2=1,b=0)=1 and Cost(w_1=1,w_2=1,b=2)=4. In this case, the learnable parameter values of w_1=1, w_2=1 and b=0 are chosen as they have a lower cost value (1<4).
Plotting out the cost function for one learnable parameter b, as seen in Fig 8, and select the value that minimize the cost; each step is called an iteration. Different values of b are specified for the cost function and at the bottom of the graph, the corresponding plane is plotted. As the cost decreases, the number of misclassified samples decreases accordingly.
![image](https://github.com/user-attachments/assets/e8674956-19fb-477d-9ed3-b5cb9be3a1df)
![image](https://github.com/user-attachments/assets/ce53e8c6-f29f-483b-81ce-72635f367bea)
![image](https://github.com/user-attachments/assets/bd3c5b11-b395-459c-9d50-c26faaeaedd4)

The jump's size is a number determined by the **learning rate**.

## CNN
A CNN is a much more complicated function that uses an image as input. Each of the small boxes below are learnable parameters called **kernels** that are applied to the image via a function called **convolution**. Each convolution outputs something called an activation map, which are represented by the larger boxes. Convolutions along with several operations are applied to each subsequent map, with each step being called a layer. Like a linear classifier, a CNN has to be trained and we can do that with pre-trained layers. Unlike a linear classifier, a CNN requires a lot of computational resources and data.
![image](https://github.com/user-attachments/assets/598dadce-b70e-47a0-99f0-33cacc6f57fd)

**Pre-trained CNNs and Transfer Learning**  
Pre-trained CNNs have been trained with large amounts of data, usually with many more classes. An example of such a dataset is ImageNet. With these CNNs, we can simply replace their last layer with a linear classifier, sometimes called a **custom layer**. Instead of having only two dimensions, this linear layer will have hundreds of dimensions. We can then train the linear layer only. This is a type of transfer learning and is an excellent way to get good results with relatively little work.
