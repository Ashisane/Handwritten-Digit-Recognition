# Handwritten Digit Recognition

### Building Neural Netowork from Scratch

I wanted to build neural networks with just intuition and raw mathematics. To startoff, I asked Claude to give me the simplest Neural Network project with a beginner hint.

It asked me to train a neural network on MNIST. 

Inital Hint: Input Layer (784) -> Hidden Layer (128) -> Ouput Layer. Network of units arranged in layers, each unit takes input -> multiplies with weights -> sum -> applies activation function. Starts with random weights and adjusts them based off error.

---

### Input Layer

Rough idea, take input image, process them, extract feature, but what are features? more importantly what is image?

So I switched to the easiest job first, that was to understand the input. Found MNIST dataset from [Hugging Face](https://huggingface.co/datasets/ylecun/mnist). I tried to dive deep into the dataset itself, and the image are 28x28 pixels. And a pixel is basically a number representing the level of grey from 0-255. Nothing more.

From the starter hint it wasn't hard to figure out that the input layer is just pixels stacked in an array. 28x28 - 784

---

### Hidden Layer

It contains 128 units. 784/128 is not an integer, so we are definitely not merging/stacking the inputs.

That leaves two options: 

1. I find 128 features from the input layer, these can be trace of the input matrix, determinant, cofactors, sum of rows of pixels. But how is this useful if a feature produces the same value for all the numbers? And isn't this just another input layer?
2. We assign random weights, 128 different matrices with 784 random weights, one for each pixel. These weights would determine how much that feature contributes. This forms our hidden layer.

Why 128? Less than 16, too few features to learn. More than a thousand, learns pixels not features.

The weights that form the hidden layer are also features like [1] but these are built to be optimized for identifying the difference between input.

---

### Output Layer

To move from hidden layer to the next output layer I actually had multiple ideas


1. Always starting from the simplest thing that comes to my mind. Sum up the hidden layer, all 128 features into one single value then normalize it before producing the mean for each label. Then the mean your sum is closest to after normalisation would be your output. This produced 22% accuracy almost random.

   ![1771563525135](image/README/1771563525135.png)

   ![1771563545260](image/README/1771563545260.png)

   This 22% accuracy was a given just the way how mean works. But this raised serious problem later on. What will be the error? How will I know which weights to update? Do I drift all the weights in same direction everytime won't that negate the effects of one another?
2. I couldn't decide on a function, since without running the entire the pipeline how would I even know which function is better? And it clicked, there is no best function, we let the neural network dictate it.

Multipying the hidden layer (128,) with 10 random weights each of length 128, signifying a digit. I let the network learn these functions from backpropogation.

Each of these 10 outputs would corrospond to one digit in any order, since it is completely upto me how I infer these and would vary only with how I use them.

---

### Activation Function - Softmax

I have 10 outputs, each corrospond to a digit. But these are just random numbers both negative and positive. I need to use them to identify a label.

I had different definitions in my mind

1) The simplest one again, we sum these output layer values. This brought a problem, I dont know what this number should be inorder to output a value. If I run the training just to find these thresholds, how will I update the weights.
2) I pick the larget one. From the output layer, i pick the argument which was largest. But this posed the same problem as before, how will I update the weights, increase, decrease, and by how much.
3) 
4) The third way was to transform these 10 numbers on the same +ve number line. And then pick the largest one. Since they are all in same direction, the error would just be distance from origin. So we convert 10 different parameters/features -> 10 different numbers, these 10 numbers would depict the probablity for each number. Deriving this function through trial and error.
   1) [1]
