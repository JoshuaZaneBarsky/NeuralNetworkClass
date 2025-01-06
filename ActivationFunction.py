import numpy as np

class Tanh:
    def activate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2
    
class ReLu:
    def activate(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
class LeakyReLu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Sigmoid:
    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.activate(x)
        return sig * (1 - sig)
    
class Softmax:
    def activate(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x):
        s = self.activate(x)
        return s * (1 - s)
    
class Swish:
    def activate(self, x):
        return x / (1 + np.exp(-x))
    
    def derivative(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)