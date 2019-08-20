Some slides are uploaded on [Google drive](https://drive.google.com/drive/u/1/folders/1VK2SUq8VSILzSPr0O2e4_8P-1ycKXbRr), more slides will be opened soon.
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
 BN: mini-batch $B =$ {$x_{1,...,m}$},  ${y_i} = BN_{\gamma ,\beta }(x_i) = \gamma {\hat x}_i + \beta$, $ {\hat x}_i = \frac{x_i - {\mu _B}}{\sqrt {\sigma _B^2 + \varepsilon}}  $

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

# 2. Convolutional Neural Networks by [Graham Taylor](https://www.gwtaylor.ca/)
* $1\times1$ kernel, 
* Stride and max pooling

# 3. DL for Image by [Angel Chang](https://angelxuanchang.github.io/)
### 3.1 Standard vision tasks
![CV tasks](https://github.com/HIT-SMC/Public-Presentations/blob/master/DLRLSS2019/Deep%20Learning/images/image1.png)
* Image classification;
* Semantic Segmentation --> fully convolutional; downsample vs. upsample, convolution vs. transpose convolution (deconvolution)
* Object Detection --> sliding window (too computational), selective search([R-CNN](https://kopernio.com/viewer?doi=10.1109/cvpr.2014.81&route=6) and [Fast R-CNN = R-CNN + feature level region proposal](https://kopernio.com/viewer?doi=10.1109/iccv.2015.169&route=6), [Faster R-CNN=Fast R-CNN + Region Proposal Network](https://kopernio.com/viewer?doi=10.1109/tpami.2016.2577031&route=6) R stands for region)
* Instance Segmentation: [Mask R-CNN](https://kopernio.com/viewer?doi=10.1109/tpami.2018.2844175&route=6) = R-CNN + Mask Network

### 3.2 From 2D pixels to 3D
* Structured representation
 * Structured representation of images: Scene Parse Tree (hierarchical description), Scene Graoh (relational description)=objects + relationships
* Visual and Language: [captioning (CNN+RNN)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7534740), [Visual Question Answering (VQA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7534740), [reasoning](https://papers.nips.cc/paper/7381-neural-symbolic-vqa-disentangling-reasoning-from-vision-and-language-understanding.pdf)
* Extending to 3D
![Image2](https://github.com/HIT-SMC/Public-Presentations/blob/master/DLRLSS2019/Deep%20Learning/images/image%202.jpg)
 * [Classification, Semantic Segmentation, Object Detection, Instance Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8099744)
 * Representation in 3D: surface (triangle mesh, hard to feed in), [multi view](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7410471),  [volumetric](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8100184), [pointcloud](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8099499)
 * PointNet, PointNet++, PointCNN, PCNN
 
### 3.3 Vision as Inverse Graphics
* Machine perception of 3D solids 3D to 2D models
* Image 2 3D scence representation
* 3D object detection: R-CNN in 3D
* 3D shape prediction
* 2D floorplan to 3D model
* Image based 3D synthesis
etc.
### Reading materials: [Standford CNN&CV course](http://cs231n.github.io/)

# 4. RNN by [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/)
### 4.1 RNN
* Sequential input/output, back-propagation through time (BPTT)
* seq2vec, seq2seq, vec2seq, seq2seq of varitional length
* Parameter sharing, $s_t=F_\theta(s_{t-1}, x_t)$, $s_t=G_t{x_t, x_{t-1},x_{t-2},...,x_2, x_1}$
* Generative RNNs, conditional distribution $P(x_t|x_{t-1},x_{t-2},...,x_2,x_1)$
* Mismatch in teacher forcing can cause 'Compounding error' --> [schedule sampling](http://papers.nips.cc/paper/5956-scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks.pdf) and [GAN](http://papers.nips.cc/paper/6099-professor-forcing-a-new-algorithm-for-training-recurrent-networks.pdf)
* Enpowered RNN: deep RNN, skip connection for creating shorter path, bidirectional RNN, Recursive Nets(tree-structured), Multidimensional RNN, [Multiplicative Integration RNNs](https://arxiv.org/pdf/1606.06630.pdf)(same computational cost, more expressive), multiscale/hierarchical RNN 
### 4.2 Problem: [Long-term dependencies with gradient descent is difficult](http://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf) --> [vanishing or exploding gradient](http://people.idsia.ch/~juergen/fundamentaldeeplearningproblem.html)
$\frac{{\partial {C_t}}}{{\partial W}} = \sum\limits_{\tau  \le t} {\frac{{\partial {C_t}}}{{\partial {a_\tau }}}\frac{{\partial {a_\tau }}}{{\partial W}}}  = \sum\limits_{\tau  \le t} {\frac{{\partial {C_t}}}{{\partial {a_t}}}\frac{{\partial {a_t}}}{{\partial {a_\tau }}}\frac{{\partial {a_\tau }}}{{\partial W}}}$, where ${\frac{{\partial {a_t}}}{{\partial {a_\tau }}}}$ becomes exponentially smaller for longer time differences when radius<1 --> vanishing gradient
* RNN tricks:[1](http://proceedings.mlr.press/v28/pascanu13.pdf) [2](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6639349)
 * Gradients clipping -- avoid exploding gradients
 * [Skip connections]() & leaky integration -- propagate further
 * Multiple time scales/ hierarchical RNNs -- propagate further
 * Momentum -- cheap 2nd order
 * Initialization -- start in right ballpark avoids exploding and vanishing
 * Sparse gradients -- symmetry breaking
 * Gradient propagation regularizer -- gradient vanishing
 * Gated self-loops -- LSTM & GRU, reduces vanishing gradient
 * Non-parametric memory (with attention mechanism) -- gradients vanishing --> memory-augmented networks
 * [Attention](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XUd1QehKguU): [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf), [Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf%20(http://Neural%20Turning%20Machines)%20), [Memory Networks](https://arxiv.org/pdf/1410.3916.pdf) , [self-attention](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [sparse attentive backtracking](http://papers.nips.cc/paper/7991-sparse-attentive-backtracking-temporal-credit-assignment-through-reminding.pdf)
 ### 4.3 Abstract concepts 
 * Content-based attention: [Consciousness Prior](https://arxiv.org/pdf/1709.08568.pdf)
 
 # 5. Video by [Greg Mori](https://www.cs.sfu.ca/~mori/#teaching)
 
 # 6. Optimization by [Jimmy Ba](https://jimmylba.github.io/)
 [DP Kingma, JL Ba. 2015. Adam: A Method For Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf) Jimmy is one of the two authors of Adam optimization method.
 ### 6.1 Random search vs. Gradient descent
 * Neural Networks <br>
 Data $(x,y)$, neural network $\hat y(x, W)$, Averaged loss: $ L = \frac{1}{N} \sum_{i=1}^{N}L(x^{(i)},y^{(i)},W)=\frac{1}{N}\sum_{i=1}^{N}L_i(W)$
 * Learning is difficult
  * Neural Networks is non-convex --> many local optimums
  * Neural Networks is not smooth --> Lipschitz continuous
  * Millions of trainable weights
 * Random search
  * perturing random vector $\Delta W \sim N(0, \mu^2I)$, evaluate the perturbed averaged loss over training examples, add the perturbation weighted by the perturbed loss to the current weights, and repeat.
  * Why work? Random search approximate finite difference with stochastic samples, each perturbation gives a directional gradient. --> inefficient however.
 * Gradient descent & back-propagation
  * $min L(W+\Delta W)$ <br>
  $ s.t. ||\Delta W||^2=\epsilon$
  * Random search, needs to do forward propagation then backward gradient; it would be more efficient to directly query gradient information --> gradient descent $\Delta W = -\nabla L$
  * Gradient descent can be inefficient under ill-conditioned curvature --> smooth gradient with moving average (Momentum)
 * Stochastic gradient descent --> improve efficiency (computing average loss of in minibatch)
 * [Natural gradient descent](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf)
  Add constraint in the Probability space using KL divergence: $s.t. D(P_W||P_{W+\Delta W})=\epsilon$
 * Fisher Information Matrix
  
  
 
 ### 6.2 Better search directions
 * Second order algorithms
 ### 6.3 "White-box" optimization methods to improve computation efficiency

# 7. NLP by [Alna Fyshe](https://www.cifar.ca/bio/alona-fyshe)
# 8. Bayesian DL by [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/)
# 9. Unsupervised Learning with [Jacobsen](https://jhjacobsen.github.io/)
# 10. Generative Model by [Ke Li](https://people.eecs.berkeley.edu/~ke.li/)

