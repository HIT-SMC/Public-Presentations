# 1. General introduction to neural networks by [Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh/index_en.html)
### 1.1 Artifical Neurons
* **Universal Approximation Theorem**: "a single hidden layer neural network with a linear output unit can approximate any continous function arbitrarily well, given enough hidden units".  <br>
However, it doesn't mean there is a learning algorithm that can find the necessary parameters. So here comes multilayer neural network and deep neural networks.
* Multilayer neural networks <br>
  * activation function: sigmoid, tanh, relu, softmax, etc.
  * flow graph: nice propagation representation
 * Tricks in training 
  * basics: empirical risj minimization, L1 and L2 regularization, loss function (classification/regression), cross-entropy, stochastic gradient descent, backpropagation, gradients of activation functions, automatic differentiation using graph
  * **Initialization**: uniformly sample weights $W_i$ from $U[-b,b]$ where $b = \frac{{\sqrt 6 }}{{\sqrt {{H_k} + {H_{k - 1}}} }}$
  * **Model selection**: abouth hyperparameters, grid search and random search, validation set
  * **Where to stop**: trade-off between trainin and validation
  * **Others**: normalization of data, decaying learning rate, minibatch, momentum, <br>
  adaptive learning rates: Adagrad, RMSProp, Adam <br>
  momentum: $ \bar \nabla _\theta ^t = \nabla _\theta L\left( \theta  \right) + \beta \bar \nabla _\theta ^{t - 1} $  <br>
  Adagrad: $ \lambda^t = \lambda^{t-1} + \left( \nabla_\theta L \left( _theta \right) \right)^2 $, $ \nabla_\theta^t = \frac{\nabla_\theta \left(L(\theta) \right)}{\sqrt{\lambda^t + \epsilon}} $

