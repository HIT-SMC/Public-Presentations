# 1. General introduction to neural networks by [Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh/index_en.html)
### 1.1 Artifical Neurons
* **Universal Approximation Theorem**: "a single hidden layer neural network with a linear output unit can approximate any continous function arbitrarily well, given enough hidden units".  <br>
However, it doesn't mean there is a learning algorithm that can find the necessary parameters. So here comes multilayer neural network and deep neural networks.
### 1.2 Multilayer neural networks <br>
  * activation function: sigmoid, tanh, relu, softmax, etc.
  * flow graph: nice propagation representation
 * Tricks in training 
  * basics: empirical risk minimization, L1 and L2 regularization, loss function (classification/regression), cross-entropy, stochastic gradient descent, backpropagation, gradients of activation functions, automatic differentiation using graph
  * **Initialization**: uniformly sample weights $W_i$ from $U[-b,b]$ where $b = \frac{{\sqrt 6 }}{{\sqrt {{H_k} + {H_{k - 1}}} }}$
  * **Model selection**: abouth hyperparameters, grid search and random search, validation set
  * **Where to stop**: trade-off between trainin and validation
  * **Others**: normalization of data, decaying learning rate, minibatch, momentum, <br>
  adaptive learning rates: Adagrad, RMSProp, Adam <br>
  **momentum**: $ \bar \nabla _\theta ^t = \nabla _\theta L\left( \theta  \right) + \beta \bar \nabla _\theta ^{t - 1} $   <br>
  **Adagrad** -- learning rates scaled by the squared root of the cumulative sum of square gradients:
  $ {\gamma ^t} = {\gamma ^{t - 1}} + {\left( {{\nabla _\theta }L\left( \theta  \right)} \right)^{\rm{2}}}$, 
  $\bar \nabla _\theta ^t = \frac{{{\nabla _\theta }L\left( \theta  \right)}}{{\sqrt {{\gamma ^t} + \varepsilon } }}$ <br>
  **RMSProp** -- scaled by exponential moving average: 
  ${\gamma ^t} = \beta {\gamma ^{t - 1}} + \left( {1 - \phi } \right){\left( {{\nabla _\theta }L\left( \theta  \right)} \right)^{\rm{2}}}$
  $\bar \nabla _\theta ^t = \frac{{{\nabla _\theta }L\left( \theta  \right)}}{{\sqrt {{\gamma ^t} + \varepsilon } }}$ <br>
  **Adam**--combines RMSProp with momentum
  * **Gradient checking** using finite difference approximation & **small set debugging**
### 1.3 Traing in Deep Learning
 * A DNN can represent certain functions more compactly e.g., Boolen functions
 * Hard to train: <br> 
 **underfitting**: i.e., gradient vanishing, saturated units block gradient propagation --> better optimization methods, use GPU
 **overfitting**: lots of parameters, induce high-variance/low-bias situation --> unsupervised pretraining (feature learning like autoencoder then finetuning), dropout etc. <br>
 [unsupervised pretraining](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf): overfits less with large capacity & underfits with small capacity
 [**Batch Normalization**](https://kopernio.com/viewer?doi=arXiv:1502.03167&route=6): both can use BN, BN attempts to normalize at the level of hidden layers besides the input level --> each unit's
 pre-activation is normalized; operates on each minibatch. <br>
 BN: mini-batch ${y_i} = B{N_{\gamma ,\beta }}\left( {{x_i}} \right)$ ,  $B =$ {$x_{1,...,m}$},  ${y_i} = B{N_{\gamma ,\beta }}({{x_i}}) = \gamma {{\hat x}_i} + \beta $
 
 
 ${y_i} = \gamma {{\hat x}_i} + \beta  \equiv BN_{\gamma ,\beta }}\left( {{x_i}} \right)$, <br>
 ${\mu _B = \frac{1}{m}\sum\limits_{i = 1}^m {{x_i}} $, ${\sigma _B = \frac{1}{m}\sum\limits_{i = 1}^m {{{\left( {{x_i} - {\mu _{B} \right)}^2}}$,${{\hat x}_i} = \frac{{x - {\mu _{B}{{\sqrt {\sigma _{\rm{{\cal B}}}^2 + \varepsilon } }}$, 

### 1.4 Learning Problems
* Supervised learning: classification, regression
* Unsupervised learning: distribution estimation, dimensionality reduction
* [Semi-supervised learning](https://www.jianshu.com/p/7e2bd0999055)
* Multitask learning: e.g., multi objects in CV
* Transfer learning
* Structured output prediction: e.g, image caption, generation, machine translation
* Domain adaption: e.g., sim2real
* One-shot learning: e.g., one-shot recognization
* zero-shot leraning: e.g., zero-shot word description
* New architectures: (1) biological inspired intuition, (2) turn an existing algorithm into Neural Network

### 1.5 Issues of ML
* [Adversarial failure](https://kopernio.com/viewer?doi=arxiv:1312.6199&route=6)
* [Non-convex problem](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization.pdf), 
[saddle points](https://kopernio.com/viewer?doi=arxiv:1412.6544&route=6), 
[intrinsic dimension](https://kopernio.com/viewer?doi=arxiv:1804.08838&route=6), 
[lotter ticket hypothesis: sparse network pruning](https://kopernio.com/viewer?doi=arxiv:1803.03635&route=6)
* Can work best when badly trained: [sharp vs. flat minima](https://kopernio.com/viewer?doi=10.1162/neco.1997.9.1.1&route=1), [etc.](https://kopernio.com/viewer?doi=arxiv:1609.04836&route=6)
* [Easily memorize](https://kopernio.com/viewer?doi=arxiv:1611.03530&route=6): model capactiy vs training algorithm
* [Easily forgete](https://kopernio.com/viewer?doi=10.1073/pnas.1611835114&route=7): lifelong learning, continual learning
* Strangely underfit/overfit: [bias/variance trade-off](https://kopernio.com/viewer?doi=arxiv:1812.11118&route=6), interpolation threshold
* Can be compressed: [distill in network](https://kopernio.com/viewer?doi=arxiv:1503.02531&route=6)
* [Influence by initialization](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf)
* [Influence by first example](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf)


 
 


