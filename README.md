# Neural Network with Student Class

This project implements a neural network using a custom `Student` class for function approximation. The network has 4 layers and has a handful of activation functions to choose from.

## Requirements

- numpy
- matplotlib

Install dependencies:
pip install numpy matplotlib

## Student Class

The `Student` class implements a neural network with:
- **4 layers** (1 input, 2 hidden, 1 output)
- **Swish** activation function (user can easily change)
- **Backpropagation** and **gradient descent** for training

Additional note:
- **Weight initialization** uses He initialization (<https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_>
)

### Methods:
1. **`forward(x)`**: Computes the output by passing `x` through the network.
2. **`backward(x, y, learning_rate)`**: Updates weights/biases via backpropagation.

## Training

- **Data**: Function to approximate (e.g., `f(x) = x^2`).
- **Batch size**: 4
- **Learning rate**: 0.0005
- **Epochs**: 1000

## Run

1. Install dependencies.
2. Ensure `Student.py` and `ActivationFunction.py` are in the same directory.
3. Run the main script to train and plot results.

This coding project was put together from the recent research guidance of Dr. Pedro Morales-Almazan of the University of California, Santa Cruz Mathematics Department.